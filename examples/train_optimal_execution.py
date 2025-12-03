#!/usr/bin/env python3
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

"""Example: Train an optimal execution policy using DiscoRL.

This script demonstrates how to:
1. Configure the optimal execution environment
2. Train a policy using Disco103 (the meta-learned update rule)
3. Compare against Actor-Critic baseline
4. Evaluate and analyze the trained policy

Usage:
    python examples/train_optimal_execution.py
    python examples/train_optimal_execution.py --num_steps 50000 --compare_baseline
"""

import argparse
from datetime import datetime
import json
import os

import jax
import numpy as np

from disco_rl.rdagent_integration import (
    ExecutionExperimentConfig,
    DiscoTrainer,
    DiscoExecutionScenario,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train optimal execution policy with DiscoRL"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10000,
        help="Number of training steps (default: 10000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="Execution horizon in steps (default: 50)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Also train Actor-Critic baseline for comparison",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_results(result, name: str = "Results"):
    """Print formatted results."""
    print(f"\n{name}:")
    print(f"  Mean shortfall:     {result.mean_shortfall:>10.4f}")
    print(f"  Std shortfall:      {result.std_shortfall:>10.4f}")
    print(f"  Completion rate:    {result.completion_rate:>10.1%}")
    print(f"  VWAP vs arrival:    {result.mean_vwap_vs_arrival:>10.4f}")
    print(f"  Mean episode len:   {result.mean_episode_length:>10.1f}")
    print(f"\n  Action distribution:")
    for action, freq in result.action_distribution.items():
        bar = "█" * int(freq * 30)
        print(f"    {action:12s}: {bar} {freq:.1%}")
    print(f"\n  Score: {result.score():.2f}")


def main():
    args = parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"execution_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print_header("DiscoRL Optimal Execution Training")
    print(f"Configuration:")
    print(f"  Training steps:  {args.num_steps}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Horizon:         {args.horizon} steps")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  Output:          {output_dir}")

    # Create configuration
    config = ExecutionExperimentConfig(
        horizon=args.horizon,
        num_training_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        # Reward shaping
        arrival_price_bonus=0.1,
        incomplete_penalty=5.0,
    )

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # ========================================================================
    # Train with Disco103
    # ========================================================================
    print_header("Training with Disco103")

    disco_trainer = DiscoTrainer(
        config,
        rng_seed=args.seed,
        use_disco=True,
    )

    print("Initializing agent...")
    disco_trainer.initialize()

    print(f"Training for {args.num_steps} steps...")
    logs = disco_trainer.train(args.num_steps)

    # Print training progress
    if logs:
        print("\nTraining progress:")
        for log in logs[::max(1, len(logs)//5)]:  # Show 5 samples
            print(f"  Step {log['step']:>6d}: mean_reward = {log['mean_reward']:.4f}")

    print("\nEvaluating trained policy...")
    disco_result = disco_trainer.evaluate()
    print_results(disco_result, "Disco103 Results")

    # Save results
    disco_trainer.save(os.path.join(output_dir, "disco103"))
    with open(os.path.join(output_dir, "disco103_results.json"), "w") as f:
        json.dump(disco_result.to_dict(), f, indent=2)

    # ========================================================================
    # Optionally compare to baseline
    # ========================================================================
    if args.compare_baseline:
        print_header("Training Actor-Critic Baseline")

        ac_trainer = DiscoTrainer(
            config,
            rng_seed=args.seed,
            use_disco=False,
        )

        ac_trainer.initialize()
        ac_trainer.train(args.num_steps)
        ac_result = ac_trainer.evaluate()
        print_results(ac_result, "Actor-Critic Results")

        # Save baseline results
        ac_trainer.save(os.path.join(output_dir, "actor_critic"))
        with open(os.path.join(output_dir, "actor_critic_results.json"), "w") as f:
            json.dump(ac_result.to_dict(), f, indent=2)

        # Print comparison
        print_header("Comparison: Disco103 vs Actor-Critic")

        shortfall_improvement = (
            (ac_result.mean_shortfall - disco_result.mean_shortfall)
            / abs(ac_result.mean_shortfall) * 100
            if ac_result.mean_shortfall != 0 else 0
        )
        completion_improvement = (
            disco_result.completion_rate - ac_result.completion_rate
        ) * 100

        print(f"Shortfall improvement:   {shortfall_improvement:>+.1f}%")
        print(f"Completion improvement:  {completion_improvement:>+.1f} pp")
        print(f"\nDisco103 score:          {disco_result.score():.2f}")
        print(f"Actor-Critic score:      {ac_result.score():.2f}")

        winner = "Disco103" if disco_result.score() > ac_result.score() else "Actor-Critic"
        print(f"\nWinner: {winner}")

    # ========================================================================
    # Summary
    # ========================================================================
    print_header("Training Complete")
    print(f"Results saved to: {output_dir}")
    print("\nFiles:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
