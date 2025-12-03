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

"""RD-Agent Integration for DiscoRL Optimal Execution.

This module provides integration between DiscoRL's learned update rules and
RD-Agent's experiment framework. The LLM in RD-Agent can evolve:
  - Environment configurations (features, reward shaping)
  - Training hyperparameters
  - Market simulation parameters

The DiscoRL agent handles the actual policy learning using Disco103.

Example RD-Agent workflow:
    1. LLM generates ExecutionExperimentConfig
    2. DiscoTrainer trains policy using Disco103
    3. Results evaluated on held-out market scenarios
    4. LLM iterates based on performance metrics
"""

from dataclasses import dataclass, field
from typing import Any
import json
import os

import jax
import jax.numpy as jnp
from ml_collections import config_dict as configdict
import numpy as np

from disco_rl import agent as disco_agent
from disco_rl.environments import optimal_execution


@dataclass
class ExecutionExperimentConfig:
    """Configuration that RD-Agent LLM can evolve.

    This dataclass defines what the LLM can tune. Each field
    represents a design decision the LLM can experiment with.
    """

    # Environment configuration
    horizon: int = 50
    order_side: int = 1  # 1=buy, -1=sell

    # Market simulation parameters
    base_spread: float = 0.01
    base_volatility: float = 0.02
    mean_reversion: float = 0.1
    momentum_factor: float = 0.3

    # Reward shaping (critical for RL performance)
    arrival_price_bonus: float = 0.1
    incomplete_penalty: float = 5.0
    wait_penalty: float = 0.001

    # Training hyperparameters
    num_training_steps: int = 100_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    trajectory_length: int = 50

    # Evaluation
    num_eval_episodes: int = 100
    eval_seeds: list = field(default_factory=lambda: [42, 123, 456])

    def to_env_config(self) -> configdict.ConfigDict:
        """Convert to DiscoRL environment config."""
        return configdict.ConfigDict(dict(
            horizon=self.horizon,
            order_side=self.order_side,
            random_seed=42,
            arrival_price_bonus=self.arrival_price_bonus,
            incomplete_penalty=self.incomplete_penalty,
        ))

    def to_dict(self) -> dict:
        """Serialize for RD-Agent storage."""
        return {
            "horizon": self.horizon,
            "order_side": self.order_side,
            "base_spread": self.base_spread,
            "base_volatility": self.base_volatility,
            "mean_reversion": self.mean_reversion,
            "momentum_factor": self.momentum_factor,
            "arrival_price_bonus": self.arrival_price_bonus,
            "incomplete_penalty": self.incomplete_penalty,
            "wait_penalty": self.wait_penalty,
            "num_training_steps": self.num_training_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "trajectory_length": self.trajectory_length,
            "num_eval_episodes": self.num_eval_episodes,
            "eval_seeds": self.eval_seeds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionExperimentConfig":
        """Deserialize from RD-Agent storage.

        Filters out unknown keys to allow for extra fields like 'use_disco'
        which are used by DiscoTrainer but not part of this config.
        """
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class ExecutionExperimentResult:
    """Results from a DiscoRL training run.

    RD-Agent uses these metrics to guide evolution.
    """

    # Primary metrics
    mean_shortfall: float  # Implementation shortfall (lower is better)
    std_shortfall: float
    completion_rate: float  # Fraction of orders fully executed

    # Detailed metrics
    mean_vwap_vs_arrival: float  # VWAP relative to arrival price
    mean_market_impact: float
    mean_episode_length: float

    # Per-action statistics
    action_distribution: dict  # How often each action was taken

    # Training metrics
    final_loss: float
    training_steps: int
    converged: bool

    def to_dict(self) -> dict:
        """Serialize for RD-Agent."""
        return {
            "mean_shortfall": self.mean_shortfall,
            "std_shortfall": self.std_shortfall,
            "completion_rate": self.completion_rate,
            "mean_vwap_vs_arrival": self.mean_vwap_vs_arrival,
            "mean_market_impact": self.mean_market_impact,
            "mean_episode_length": self.mean_episode_length,
            "action_distribution": self.action_distribution,
            "final_loss": self.final_loss,
            "training_steps": self.training_steps,
            "converged": self.converged,
        }

    def score(self) -> float:
        """Single scalar score for RD-Agent optimization.

        Higher is better. Combines shortfall, completion, and consistency.
        """
        # Invert shortfall (we want to minimize it)
        shortfall_score = -self.mean_shortfall * 100

        # Completion bonus
        completion_score = self.completion_rate * 10

        # Consistency bonus (low variance)
        consistency_score = -self.std_shortfall * 50

        return shortfall_score + completion_score + consistency_score


class DiscoTrainer:
    """Trains optimal execution policies using DiscoRL.

    This class wraps the DiscoRL agent training loop and provides
    an interface suitable for RD-Agent integration.
    """

    def __init__(
        self,
        config: ExecutionExperimentConfig,
        rng_seed: int = 42,
        use_disco: bool = True,
    ):
        """Initialize trainer.

        Args:
            config: Experiment configuration
            rng_seed: Random seed for reproducibility
            use_disco: If True, use Disco103. If False, use Actor-Critic baseline.
        """
        self.config = config
        self.rng = jax.random.PRNGKey(rng_seed)
        self.use_disco = use_disco

        # Create environment
        self.env_config = config.to_env_config()
        self.env = optimal_execution.OptimalExecutionEnvironment(
            batch_size=config.batch_size,
            env_settings=self.env_config,
        )

        # Create agent
        if use_disco:
            agent_settings = disco_agent.get_settings_disco()
        else:
            agent_settings = disco_agent.get_settings_actor_critic()

        # Override learning rate from config
        agent_settings.learning_rate = config.learning_rate

        self.agent = disco_agent.Agent(
            single_observation_spec=self.env.single_observation_spec(),
            single_action_spec=self.env.single_action_spec(),
            agent_settings=agent_settings,
            batch_axis_name=None,
        )

        # Training state
        self.learner_state = None
        self.actor_state = None
        self.env_state = None
        self.training_logs = []

    def initialize(self):
        """Initialize agent and environment states."""
        self.rng, init_rng, env_rng = jax.random.split(self.rng, 3)

        # Initialize environment
        self.env_state, env_timestep = self.env.reset(env_rng)

        # Initialize agent
        self.learner_state = self.agent.initial_learner_state(init_rng)
        self.actor_state = self.agent.initial_actor_state()

    def train(self, num_steps: int | None = None) -> list[dict]:
        """Run training loop.

        Args:
            num_steps: Number of training steps. If None, uses config value.

        Returns:
            List of training logs (one per step).
        """
        if self.learner_state is None:
            self.initialize()

        num_steps = num_steps or self.config.num_training_steps
        logs = []

        for step in range(num_steps):
            self.rng, step_rng = jax.random.split(self.rng)

            # Collect trajectory
            rollout, self.actor_state, self.env_state = self._collect_trajectory(
                step_rng
            )

            # Learner step
            self.learner_state, _, step_log = self.agent.learner_step(
                self.learner_state,
                rollout,
            )

            # Log periodically
            if step % 1000 == 0:
                log_entry = {
                    "step": step,
                    "mean_reward": float(jnp.mean(rollout.rewards)),
                }
                logs.append(log_entry)

        self.training_logs = logs
        return logs

    def _collect_trajectory(self, rng):
        """Collect a trajectory of experience."""
        trajectory_length = self.config.trajectory_length

        timesteps = []
        actor_state = self.actor_state
        env_state = self.env_state

        for t in range(trajectory_length):
            rng, action_rng = jax.random.split(rng)

            # Get current observation
            env_state, env_timestep = self.env.step(env_state, None)  # Get obs

            # Actor step
            actor_timestep, actor_state = self.agent.actor_step(
                self.learner_state,
                actor_state,
                env_timestep,
                action_rng,
            )

            # Environment step
            env_state, env_timestep = self.env.step(env_state, actor_timestep.actions)

            timesteps.append(actor_timestep)

        # Stack into rollout
        from disco_rl import types
        rollout = types.ActorRollout(
            observations=jax.tree.map(lambda *xs: jnp.stack(xs), *[t.observations for t in timesteps]),
            actions=jnp.stack([t.actions for t in timesteps]),
            rewards=jnp.stack([t.rewards for t in timesteps]),
            discounts=jnp.stack([t.discounts for t in timesteps]),
            agent_outs={k: jnp.stack([t.agent_outs[k] for t in timesteps]) for k in timesteps[0].agent_outs},
            states=jax.tree.map(lambda *xs: jnp.stack(xs), *[t.states for t in timesteps]),
            logits=jnp.stack([t.logits for t in timesteps]),
        )

        return rollout, actor_state, env_state

    def evaluate(self, num_episodes: int | None = None) -> ExecutionExperimentResult:
        """Evaluate trained policy.

        Args:
            num_episodes: Number of evaluation episodes.

        Returns:
            ExecutionExperimentResult with performance metrics.
        """
        num_episodes = num_episodes or self.config.num_eval_episodes

        shortfalls = []
        completion_rates = []
        vwap_vs_arrivals = []
        episode_lengths = []
        action_counts = {i: 0 for i in range(optimal_execution.NUM_ACTIONS)}
        total_actions = 0

        for seed in self.config.eval_seeds:
            eval_rng = jax.random.PRNGKey(seed)

            # Create fresh eval environment
            eval_env_config = self.env_config.copy_and_resolve_references()
            eval_env_config.random_seed = seed
            eval_env = optimal_execution.SingleStreamOptimalExecution(eval_env_config)

            for _ in range(num_episodes // len(self.config.eval_seeds)):
                timestep = eval_env.reset()
                episode_reward = 0.0
                episode_length = 0

                # Reset actor state for episode
                actor_state = self.agent.initial_actor_state()

                while not timestep.last():
                    eval_rng, action_rng = jax.random.split(eval_rng)

                    # Get action from policy
                    obs = timestep.observation[np.newaxis, ...]  # Add batch dim
                    env_ts = types.EnvironmentTimestep(
                        observation={"obs": obs},
                        step_type=jnp.array([timestep.step_type]),
                        reward=jnp.array([timestep.reward or 0.0]),
                    )

                    actor_timestep, actor_state = self.agent.actor_step(
                        self.learner_state,
                        actor_state,
                        env_ts,
                        action_rng,
                    )

                    action = int(actor_timestep.actions[0])
                    action_counts[action] += 1
                    total_actions += 1

                    timestep = eval_env.step(action)
                    episode_reward += timestep.reward
                    episode_length += 1

                # Record metrics
                shortfalls.append(-episode_reward)  # Convert reward to shortfall
                completion_rates.append(1.0 if eval_env._remaining_qty < 0.01 else 0.0)
                if eval_env._executed_qty > 0:
                    vwap_vs_arrivals.append(
                        (eval_env._vwap - eval_env._arrival_price) / eval_env._arrival_price
                    )
                episode_lengths.append(episode_length)

        # Compute action distribution
        action_dist = {
            optimal_execution.ACTIONS[k]["name"]: v / total_actions
            for k, v in action_counts.items()
        }

        return ExecutionExperimentResult(
            mean_shortfall=float(np.mean(shortfalls)),
            std_shortfall=float(np.std(shortfalls)),
            completion_rate=float(np.mean(completion_rates)),
            mean_vwap_vs_arrival=float(np.mean(vwap_vs_arrivals)) if vwap_vs_arrivals else 0.0,
            mean_market_impact=0.0,  # TODO: track this in env
            mean_episode_length=float(np.mean(episode_lengths)),
            action_distribution=action_dist,
            final_loss=self.training_logs[-1].get("loss", 0.0) if self.training_logs else 0.0,
            training_steps=len(self.training_logs) * 1000,
            converged=True,  # TODO: implement convergence check
        )

    def save(self, path: str):
        """Save trained model and config."""
        os.makedirs(path, exist_ok=True)

        # Save config
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save learner state (params)
        # Note: In practice, use proper JAX serialization
        # This is simplified for illustration
        np.savez(
            os.path.join(path, "params.npz"),
            **jax.tree.map(np.array, self.learner_state.params)
        )

    def load(self, path: str):
        """Load trained model."""
        # Load config
        with open(os.path.join(path, "config.json"), "r") as f:
            config_dict = json.load(f)
            self.config = ExecutionExperimentConfig.from_dict(config_dict)

        # Note: Full loading requires proper JAX deserialization


# ============================================================================
# RD-Agent Scenario Interface
# ============================================================================

class DiscoExecutionScenario:
    """RD-Agent scenario for DiscoRL optimal execution.

    This class provides the interface that RD-Agent expects for
    running experiments and evolving configurations.

    Usage in RD-Agent:
        scenario = DiscoExecutionScenario()

        # LLM generates config
        config = scenario.generate_initial_config()

        # Run experiment
        result = scenario.run_experiment(config)

        # LLM evolves based on result
        new_config = scenario.evolve_config(config, result)
    """

    def __init__(self, workspace_path: str = "./disco_workspace"):
        self.workspace_path = workspace_path
        os.makedirs(workspace_path, exist_ok=True)

    def generate_initial_config(self) -> ExecutionExperimentConfig:
        """Generate default starting configuration."""
        return ExecutionExperimentConfig()

    def run_experiment(
        self,
        config: ExecutionExperimentConfig,
        experiment_id: str = "exp_001",
    ) -> ExecutionExperimentResult:
        """Run a single experiment with given configuration.

        Args:
            config: Experiment configuration
            experiment_id: Unique identifier for this experiment

        Returns:
            Experiment results
        """
        # Create trainer
        trainer = DiscoTrainer(config)

        # Train
        trainer.train()

        # Evaluate
        result = trainer.evaluate()

        # Save artifacts
        exp_path = os.path.join(self.workspace_path, experiment_id)
        trainer.save(exp_path)

        # Save results
        with open(os.path.join(exp_path, "results.json"), "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return result

    def compare_to_baseline(
        self,
        config: ExecutionExperimentConfig,
    ) -> dict:
        """Compare Disco103 to Actor-Critic baseline.

        Returns:
            Dict with disco_result, baseline_result, and improvement metrics.
        """
        # Train with Disco103
        disco_trainer = DiscoTrainer(config, use_disco=True)
        disco_trainer.train()
        disco_result = disco_trainer.evaluate()

        # Train with Actor-Critic
        ac_trainer = DiscoTrainer(config, use_disco=False)
        ac_trainer.train()
        ac_result = ac_trainer.evaluate()

        return {
            "disco": disco_result.to_dict(),
            "actor_critic": ac_result.to_dict(),
            "shortfall_improvement": (
                ac_result.mean_shortfall - disco_result.mean_shortfall
            ) / ac_result.mean_shortfall * 100,
            "completion_improvement": (
                disco_result.completion_rate - ac_result.completion_rate
            ) * 100,
        }


# ============================================================================
# LLM Prompt Templates for RD-Agent
# ============================================================================

EXPERIMENT_GENERATION_PROMPT = """
You are designing an optimal execution experiment using DiscoRL.

The goal is to train an RL agent that executes large orders while minimizing
market impact and slippage. The agent uses Disco103, a meta-learned update rule.

Current best configuration:
{current_config}

Current best results:
- Mean shortfall: {mean_shortfall:.4f}
- Completion rate: {completion_rate:.2%}
- Action distribution: {action_distribution}

Based on these results, propose a new configuration to improve performance.
Focus on:
1. Reward shaping parameters (arrival_price_bonus, incomplete_penalty)
2. Training hyperparameters (learning_rate, batch_size)
3. Environment settings (horizon for urgency)

Output your configuration as JSON.
"""

RESULT_ANALYSIS_PROMPT = """
Analyze these optimal execution experiment results:

Configuration:
{config}

Results:
- Mean shortfall: {mean_shortfall:.4f} (lower is better)
- Std shortfall: {std_shortfall:.4f}
- Completion rate: {completion_rate:.2%}
- VWAP vs arrival: {vwap_vs_arrival:.4f}
- Action distribution: {action_distribution}

Provide:
1. Assessment of the policy behavior
2. Likely causes of any issues
3. Specific suggestions for improvement
"""
