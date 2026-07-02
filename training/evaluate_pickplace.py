"""Evaluate a trained pick and place model.

Usage:
    python training/evaluate_pickplace.py [--episodes 50] [--render]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from sb3_contrib import TQC
from env.custom_env import CustomHouseholdPandaEnv

# Register if not already registered
try:
    register(
        id="NeuroGenesisPickPlace-v0",
        entry_point="env.custom_env:CustomHouseholdPandaEnv",
        max_episode_steps=100,
    )
except gym.error.Error:
    pass  # already registered


def main():
    parser = argparse.ArgumentParser(description="Evaluate pick and place policy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="Render visually")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("models", "pickplace", "best", "best_model.zip"),
    )
    args = parser.parse_args()

    render_mode = "human" if args.render else "rgb_array"
    env = gym.make("NeuroGenesisPickPlace-v0", render_mode=render_mode)

    print(f"Loading model from: {args.model_path}")
    try:
        model = TQC.load(args.model_path, env=env)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {args.model_path}.")
        print("Please ensure training has completed and saved a model.")
        sys.exit(1)

    successes = []
    rewards_list = []
    lengths = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if args.render:
                time.sleep(0.05)

        success = info.get("is_success", False)
        successes.append(float(success))
        rewards_list.append(total_reward)
        lengths.append(steps)

        status = "[PASS]" if success else "[FAIL]"
        print(f"  Episode {ep+1:3d}/{args.episodes}  {status}  reward={total_reward:.3f}  steps={steps}")

    env.close()

    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    print(f"  Episodes       : {args.episodes}")
    print(f"  Success Rate   : {np.mean(successes)*100:.1f}%")
    print(f"  Mean Reward    : {np.mean(rewards_list):.3f} ± {np.std(rewards_list):.3f}")
    print(f"  Mean Length    : {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
