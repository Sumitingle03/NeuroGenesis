"""Train the NeuroGenesis mobile manipulator to reach a target.

Usage:
    python training/train_reach.py [--timesteps 500000]
"""

import argparse
import os
import sys

# Ensure project root is on the path so `env.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium.envs.registration import register
from sb3_contrib import TQC
from stable_baselines3.common.buffers import DictReplayBuffer
from training.callbacks import make_reach_callbacks
from env.reach_env import NeuroGenesisReachEnv

# ---------------------------------------------------------------------------
#  Register the environment
# ---------------------------------------------------------------------------
register(
    id="NeuroGenesisReach-v0",
    entry_point="env.reach_env:NeuroGenesisReachEnv",
    max_episode_steps=100,
)


def make_env(render_mode="rgb_array"):
    """Create the reach environment."""
    return gym.make("NeuroGenesisReach-v0", render_mode=render_mode)


def main():
    parser = argparse.ArgumentParser(description="Train reach policy")
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total training timesteps (default: 500k)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  NeuroGenesis — Phase 2a: Reach / Navigate Training")
    print("=" * 60)

    # ---- Environments ----
    train_env = make_env()
    eval_env = make_env()

    # ---- Model ----
    model = TQC(
        policy="MultiInputPolicy",
        env=train_env,
        batch_size=2048,
        gamma=0.95,
        learning_rate=1e-3,
        tau=0.05,
        policy_kwargs=dict(
            net_arch=[512, 512, 512],
            n_critics=2,
        ),
        replay_buffer_class=DictReplayBuffer,
        buffer_size=1_000_000,
        learning_starts=1_000,
        verbose=1,
        tensorboard_log="logs/reach_tensorboard",
    )

    print(f"\nAction space : {train_env.action_space}")
    print(f"Obs space    : {train_env.observation_space}")
    print(f"Total steps  : {args.timesteps:,}")
    print()

    # ---- Callbacks ----
    callbacks = make_reach_callbacks(eval_env)

    # ---- Train ----
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        log_interval=10,
    )

    # ---- Save final model ----
    final_path = os.path.join("models", "reach", "final")
    os.makedirs(final_path, exist_ok=True)
    model.save(os.path.join(final_path, "tqc_reach_final"))
    print(f"\nFinal model saved to {final_path}")

    train_env.close()
    eval_env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
