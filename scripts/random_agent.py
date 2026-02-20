# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate environment with random action agent and log metrics."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent baseline for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (default: 1).")
parser.add_argument("--task", type=str, default="Template-Robotdogstanding-v0", help="Name of the task.")
parser.add_argument("--num_episodes", type=int, default=20, help="Number of episodes to run (default: 20).")
parser.add_argument("--step_log_interval", type=int, default=10, help="Log step data every N steps (default: 10).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import json
import os
import torch
from datetime import datetime
from pathlib import Path

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import robotDogStanding.tasks  # noqa: F401

# Import StandingSuccessMetric directly from the metrics module
# to avoid triggering the full package import
from robotDogStanding.utils.metrics import StandingSuccessMetric




def setup_logging(base_dir: str = "outputs/logs/random_policy") -> dict:
    """Setup logging directory and files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create CSV files with headers
    episode_csv_path = log_dir / "episode_metrics.csv"
    with open(episode_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "total_reward",
            "episode_length",
            "success",
            "final_height",
            "final_roll",
            "final_pitch",
            "avg_height",
            "avg_abs_roll",
            "avg_abs_pitch",
        ])

    step_csv_path = log_dir / "step_log.csv"
    with open(step_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "step",
            "height",
            "roll",
            "pitch",
            "yaw",
            "reward",
            "done",
            "success_condition_met",
        ])

    return {
        "log_dir": log_dir,
        "episode_csv": episode_csv_path,
        "step_csv": step_csv_path,
    }


def log_step_data(csv_path: Path, episode: int, step: int, metrics: dict, reward: float, done: bool):
    """Log step-level data to CSV."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode,
            step,
            metrics["height"].item(),
            metrics["roll"].item(),
            metrics["pitch"].item(),
            metrics["yaw"].item(),
            reward,
            int(done),
            int(metrics.get("conditions_met", False)),
        ])


def log_episode_data(csv_path: Path, episode: int, episode_metrics: dict):
    """Log episode-level data to CSV."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode,
            episode_metrics["total_reward"],
            episode_metrics["episode_length"],
            int(episode_metrics["success"]),
            episode_metrics["final_height"],
            episode_metrics["final_roll"],
            episode_metrics["final_pitch"],
            episode_metrics["avg_height"],
            episode_metrics["avg_abs_roll"],
            episode_metrics["avg_abs_pitch"],
        ])


def save_summary(log_dir: Path, all_episodes: list):
    """Save summary statistics to JSON."""
    total_rewards = [ep["total_reward"] for ep in all_episodes]
    episode_lengths = [ep["episode_length"] for ep in all_episodes]
    successes = [ep["success"] for ep in all_episodes]

    summary = {
        "num_episodes": len(all_episodes),
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "avg_return": sum(total_rewards) / len(total_rewards) if total_rewards else 0.0,
        "avg_episode_length": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0,
        "std_return": torch.tensor(total_rewards).std().item() if len(total_rewards) > 1 else 0.0,
        "min_return": min(total_rewards) if total_rewards else 0.0,
        "max_return": max(total_rewards) if total_rewards else 0.0,
    }

    summary_path = log_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    """Random actions agent with Isaac Lab environment."""
    # Ensure num_envs is 1 for episodic evaluation
    if args_cli.num_envs is None:
        args_cli.num_envs = 1
    elif args_cli.num_envs != 1:
        print(f"[WARNING]: num_envs set to {args_cli.num_envs}, but for baseline evaluation using num_envs=1")
        args_cli.num_envs = 1

    # Setup logging
    log_paths = setup_logging()
    print(f"[INFO]: Logging to {log_paths['log_dir']}")

    # create environment configuration (same way as train.py)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info
    print(f"[INFO]: Task: {args_cli.task}")
    print(f"[INFO]: Num episodes: {args_cli.num_episodes}")
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Initialize success metric tracker
    success_metric = StandingSuccessMetric(env.unwrapped)

    # Storage for all episodes
    all_episodes = []

    # Run episodes
    episode_count = 0
    while episode_count < args_cli.num_episodes and simulation_app.is_running():
        # Reset environment
        obs, _ = env.reset()
        success_metric.reset()

        # Episode tracking
        episode_reward = 0.0
        episode_length = 0
        episode_heights = []
        episode_rolls = []
        episode_pitches = []

        done = False
        step_count = 0

        print(f"\n[INFO]: Starting episode {episode_count + 1}/{args_cli.num_episodes}")

        while not done and simulation_app.is_running():
            # with torch.inference_mode():
                with torch.no_grad():
                # Sample random actions uniformly from action space
                # Action space is typically normalized to [-1, 1]
                    actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1

                # Step environment
                obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated[0].item() or truncated[0].item()

                # Get metrics from success tracker
                conditions_met, metrics = success_metric.compute_conditions()
                success_metric.update(terminated | truncated)

                # Accumulate episode stats
                episode_reward += reward[0].item()
                episode_length += 1
                episode_heights.append(metrics["height"][0].item())
                episode_rolls.append(metrics["roll"][0].item())
                episode_pitches.append(metrics["pitch"][0].item())

                # Log step data periodically
                if step_count % args_cli.step_log_interval == 0:
                    step_metrics = {
                        "height": metrics["height"][0],
                        "roll": metrics["roll"][0],
                        "pitch": metrics["pitch"][0],
                        "yaw": metrics["yaw"][0],
                        "conditions_met": conditions_met[0].item(),
                    }
                    log_step_data(
                        log_paths["step_csv"],
                        episode_count + 1,
                        step_count,
                        step_metrics,
                        reward[0].item(),
                        done,
                    )

                step_count += 1

        # Episode finished
        episode_success = success_metric.success_achieved[0].item()

        episode_data = {
            "total_reward": episode_reward,
            "episode_length": episode_length,
            "success": episode_success,
            "final_height": episode_heights[-1] if episode_heights else 0.0,
            "final_roll": episode_rolls[-1] if episode_rolls else 0.0,
            "final_pitch": episode_pitches[-1] if episode_pitches else 0.0,
            "avg_height": sum(episode_heights) / len(episode_heights) if episode_heights else 0.0,
            "avg_abs_roll": sum(abs(r) for r in episode_rolls) / len(episode_rolls) if episode_rolls else 0.0,
            "avg_abs_pitch": sum(abs(p) for p in episode_pitches) / len(episode_pitches) if episode_pitches else 0.0,
        }

        all_episodes.append(episode_data)
        log_episode_data(log_paths["episode_csv"], episode_count + 1, episode_data)

        print(f"[INFO]: Episode {episode_count + 1} complete:")
        print(f"  - Reward: {episode_reward:.2f}")
        print(f"  - Length: {episode_length}")
        print(f"  - Success: {episode_success}")
        print(f"  - Final height: {episode_data['final_height']:.3f}m")

        episode_count += 1

    # Save summary
    summary = save_summary(log_paths["log_dir"], all_episodes)
    print("\n" + "=" * 60)
    print("RANDOM POLICY BASELINE SUMMARY")
    print("=" * 60)
    print(f"Episodes completed: {summary['num_episodes']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Average return: {summary['avg_return']:.2f} ± {summary['std_return']:.2f}")
    print(f"Average episode length: {summary['avg_episode_length']:.1f}")
    print(f"Return range: [{summary['min_return']:.2f}, {summary['max_return']:.2f}]")
    print(f"\nLogs saved to: {log_paths['log_dir']}")
    print("=" * 60)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
