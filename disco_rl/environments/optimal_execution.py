# Copyright 2025 Trevor Morgan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Optimal Execution Environment for DiscoRL.

This environment simulates the optimal execution problem: executing a large
order over a fixed time horizon while minimizing market impact and slippage.

State space:
  - remaining_qty: Fraction of order remaining [0, 1]
  - time_remaining: Fraction of time remaining [0, 1]
  - spread: Current bid-ask spread (normalized)
  - volatility: Recent price volatility (normalized)
  - momentum: Recent price momentum (normalized)
  - imbalance: Order book imbalance [-1, 1]
  - vwap_deviation: Current VWAP vs target (normalized)

Action space (discrete):
  - 0: WAIT - No execution this step
  - 1: PASSIVE - Post limit order (lower cost, execution risk)
  - 2: MODERATE - Execute 10% of remaining
  - 3: AGGRESSIVE - Execute 25% of remaining
  - 4: URGENT - Cross spread, execute 50% of remaining

Reward:
  - Negative implementation shortfall (arrival price - execution price)
  - Penalty for market impact
  - Penalty for incomplete execution at deadline
"""

import dm_env
from dm_env import specs
from ml_collections import config_dict as configdict
import numpy as np

from disco_rl.environments.wrappers import batched_env
from disco_rl.environments.wrappers import single_stream_env


# Action definitions
ACTIONS = {
    0: {"name": "WAIT", "exec_frac": 0.0, "impact_mult": 0.0},
    1: {"name": "PASSIVE", "exec_frac": 0.05, "impact_mult": 0.2},
    2: {"name": "MODERATE", "exec_frac": 0.10, "impact_mult": 0.5},
    3: {"name": "AGGRESSIVE", "exec_frac": 0.25, "impact_mult": 1.0},
    4: {"name": "URGENT", "exec_frac": 0.50, "impact_mult": 2.0},
}
NUM_ACTIONS = len(ACTIONS)


class MarketSimulator:
    """Simple market microstructure simulator.

    Generates realistic market conditions including:
    - Price dynamics with mean reversion and momentum
    - Time-varying volatility
    - Order book imbalance
    - Bid-ask spread dynamics
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        initial_price: float = 100.0,
        base_spread: float = 0.01,
        base_volatility: float = 0.02,
        mean_reversion: float = 0.1,
        momentum_factor: float = 0.3,
    ):
        self.rng = rng
        self.initial_price = initial_price
        self.price = initial_price
        self.base_spread = base_spread
        self.base_volatility = base_volatility
        self.mean_reversion = mean_reversion
        self.momentum_factor = momentum_factor

        # State variables
        self.volatility = base_volatility
        self.spread = base_spread
        self.momentum = 0.0
        self.imbalance = 0.0
        self._price_history = [initial_price]

    def reset(self):
        """Reset market to initial state."""
        self.price = self.initial_price
        self.volatility = self.base_volatility
        self.spread = self.base_spread
        self.momentum = 0.0
        self.imbalance = 0.0
        self._price_history = [self.initial_price]

    def step(self) -> dict:
        """Advance market by one timestep."""
        # Update volatility (mean-reverting)
        vol_shock = self.rng.normal(0, 0.1)
        self.volatility = np.clip(
            self.volatility + 0.1 * (self.base_volatility - self.volatility) + 0.01 * vol_shock,
            0.005,
            0.1
        )

        # Update momentum
        momentum_shock = self.rng.normal(0, 0.1)
        self.momentum = 0.8 * self.momentum + 0.2 * momentum_shock

        # Price update: random walk + momentum + mean reversion
        price_return = (
            self.rng.normal(0, self.volatility)
            + self.momentum_factor * self.momentum
            - self.mean_reversion * (self.price - self.initial_price) / self.initial_price
        )
        self.price *= (1 + price_return)
        self._price_history.append(self.price)

        # Update spread (wider in high vol)
        self.spread = self.base_spread * (1 + 2 * (self.volatility / self.base_volatility - 1))

        # Update order book imbalance (random)
        imbalance_shock = self.rng.normal(0, 0.3)
        self.imbalance = np.clip(0.7 * self.imbalance + 0.3 * imbalance_shock, -1, 1)

        return {
            "price": self.price,
            "spread": self.spread,
            "volatility": self.volatility,
            "momentum": self.momentum,
            "imbalance": self.imbalance,
        }

    def execute(self, qty_frac: float, impact_mult: float) -> tuple[float, float]:
        """Execute a trade and return (execution_price, market_impact).

        Args:
            qty_frac: Fraction of total order to execute
            impact_mult: Impact multiplier based on urgency

        Returns:
            execution_price: Price at which trade executed
            market_impact: Permanent price impact
        """
        if qty_frac <= 0:
            return self.price, 0.0

        # Temporary impact (slippage)
        temp_impact = impact_mult * self.spread * 0.5 * np.sqrt(qty_frac)

        # Permanent impact
        perm_impact = impact_mult * self.volatility * qty_frac * 0.1

        # Execution price (mid + slippage)
        exec_price = self.price * (1 + temp_impact)

        # Move price permanently
        self.price *= (1 + perm_impact)

        return exec_price, perm_impact


class SingleStreamOptimalExecution:
    """Optimal execution environment for a single order."""

    def __init__(self, env_settings: configdict.ConfigDict):
        """Initialize the environment.

        Args:
            env_settings: Configuration containing:
                - horizon: Number of steps to complete execution
                - order_side: 1 for buy, -1 for sell
                - random_seed: RNG seed
                - arrival_price_bonus: Reward shaping for beating arrival
                - incomplete_penalty: Penalty multiplier for leftover quantity
        """
        self._horizon = env_settings.horizon
        self._order_side = env_settings.get("order_side", 1)  # 1=buy, -1=sell
        self._arrival_price_bonus = env_settings.get("arrival_price_bonus", 0.1)
        self._incomplete_penalty = env_settings.get("incomplete_penalty", 5.0)

        self._rng = np.random.RandomState(env_settings.random_seed)
        self._market = MarketSimulator(self._rng)

        # Episode state
        self._step_count = 0
        self._remaining_qty = 1.0
        self._arrival_price = 0.0
        self._vwap = 0.0
        self._executed_qty = 0.0
        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
        """Reset to start a new execution episode."""
        self._reset_next_step = False
        self._step_count = 0
        self._remaining_qty = 1.0
        self._executed_qty = 0.0

        self._market.reset()
        self._arrival_price = self._market.price
        self._vwap = 0.0

        return dm_env.restart(self._observation())

    def step(self, action: int) -> dm_env.TimeStep:
        """Execute one step in the environment."""
        if self._reset_next_step:
            return self.reset()

        action_config = ACTIONS[action]
        exec_frac = action_config["exec_frac"]
        impact_mult = action_config["impact_mult"]

        # Calculate actual execution quantity
        actual_exec = min(exec_frac * self._remaining_qty, self._remaining_qty)

        # Execute trade
        exec_price, _ = self._market.execute(actual_exec, impact_mult)

        # Update VWAP
        if actual_exec > 0:
            old_total = self._vwap * self._executed_qty
            self._executed_qty += actual_exec
            self._vwap = (old_total + exec_price * actual_exec) / self._executed_qty

        # Update remaining
        self._remaining_qty -= actual_exec
        self._step_count += 1

        # Advance market
        self._market.step()

        # Calculate reward
        reward = self._compute_reward(exec_price, actual_exec)

        # Check termination
        done = (self._step_count >= self._horizon) or (self._remaining_qty <= 1e-6)

        if done:
            # Add terminal penalty for incomplete execution
            if self._remaining_qty > 1e-6:
                reward -= self._incomplete_penalty * self._remaining_qty
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=self._observation())
        else:
            return dm_env.transition(reward=reward, observation=self._observation())

    def _compute_reward(self, exec_price: float, exec_qty: float) -> float:
        """Compute step reward based on execution quality."""
        if exec_qty <= 0:
            # Small penalty for waiting (opportunity cost)
            return -0.001

        # Implementation shortfall: how much worse than arrival price
        # For buys: negative if exec_price > arrival (paid more)
        # For sells: negative if exec_price < arrival (received less)
        shortfall = self._order_side * (self._arrival_price - exec_price) / self._arrival_price

        # Scale by quantity executed
        reward = shortfall * exec_qty

        # Bonus for beating arrival price
        if shortfall > 0:
            reward += self._arrival_price_bonus * exec_qty

        return reward

    def _observation(self) -> np.ndarray:
        """Return current observation vector."""
        time_remaining = 1.0 - (self._step_count / self._horizon)

        # Normalize market features
        spread_norm = self._market.spread / self._market.base_spread - 1.0
        vol_norm = self._market.volatility / self._market.base_volatility - 1.0

        # VWAP deviation from arrival
        if self._executed_qty > 0:
            vwap_dev = (self._vwap - self._arrival_price) / self._arrival_price
        else:
            vwap_dev = 0.0

        obs = np.array([
            self._remaining_qty,           # [0, 1]
            time_remaining,                # [0, 1]
            np.clip(spread_norm, -1, 1),   # normalized spread
            np.clip(vol_norm, -1, 1),      # normalized volatility
            np.clip(self._market.momentum, -1, 1),  # momentum
            self._market.imbalance,        # order book imbalance [-1, 1]
            np.clip(vwap_dev * 10, -1, 1), # VWAP deviation (scaled)
        ], dtype=np.float32)

        return obs

    def observation_spec(self) -> specs.Array:
        """Return observation specification."""
        return specs.Array(
            shape=(7,),
            dtype=np.float32,
            name="execution_state",
        )

    def action_spec(self) -> specs.BoundedArray:
        """Return action specification."""
        return specs.BoundedArray((), np.int32, 0, NUM_ACTIONS - 1)


class OptimalExecutionEnvironment(batched_env.BatchedSingleStreamEnvironment):
    """Batched optimal execution environment for DiscoRL."""

    def __init__(
        self,
        batch_size: int,
        env_settings: configdict.ConfigDict,
    ) -> None:
        def _single_stream_execution(
            env_settings: configdict.ConfigDict,
        ) -> single_stream_env.SingleStreamEnv:
            return single_stream_env.SingleStreamEnv(
                env=SingleStreamOptimalExecution(env_settings)
            )

        super().__init__(
            _single_stream_execution,
            batch_size,
            env_settings,
        )


def get_config() -> configdict.ConfigDict:
    """Returns default config for OptimalExecutionEnvironment."""
    return configdict.ConfigDict(
        dict(
            horizon=50,           # 50 steps to complete execution
            order_side=1,         # 1=buy, -1=sell
            random_seed=42,
            arrival_price_bonus=0.1,
            incomplete_penalty=5.0,
        )
    )


# Convenience presets for different execution scenarios
def get_config_fast() -> configdict.ConfigDict:
    """Fast execution scenario (urgent order)."""
    config = get_config()
    config.horizon = 20
    config.incomplete_penalty = 10.0
    return config


def get_config_patient() -> configdict.ConfigDict:
    """Patient execution scenario (can wait for good prices)."""
    config = get_config()
    config.horizon = 100
    config.incomplete_penalty = 2.0
    return config
