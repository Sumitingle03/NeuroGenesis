"""Train the NeuroGenesis mobile manipulator for Pick-and-Place.

Usage:
    python training/train_pickplace.py [--timesteps 1000000]
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
from env.custom_env import CustomHouseholdPandaEnv

# ---------------------------------------------------------------------------
#  Register the environment
# ---------------------------------------------------------------------------
register(
    id="NeuroGenesisPickPlace-v0",
    entry_point="env.custom_env:CustomHouseholdPandaEnv",
    max_episode_steps=100,
)


def make_env(render_mode="rgb_array"):
    """Create the pick and place environment."""
    return gym.make("NeuroGenesisPickPlace-v0", render_mode=render_mode)


def main():
    parser = argparse.ArgumentParser(description="Train pick and place policy")
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000,
        help="Total training timesteps (default: 1M)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  NeuroGenesis — Phase 2b: Pick & Place Training")
    print("=" * 60)

    # ---- Environments ----
    train_env = make_env()
    eval_env = make_env()

    # ---- Model ----
    # Pre-trained weights from reach could theoretically be loaded here,
    # but starting from scratch with TQC+HER is standard and effective
    # for picking tasks.
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
        tensorboard_log="logs/pickplace_tensorboard",
    )

    print(f"\nAction space : {train_env.action_space}")
    print(f"Obs space    : {train_env.observation_space}")
    print(f"Total steps  : {args.timesteps:,}")
    print()

    # ---- Callbacks ----
    # Reusing the callback builder but changing log dirs
    import training.callbacks
    def make_pickplace_callbacks(eval_env):
        import os
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
        os.makedirs(os.path.join("models", "pickplace"), exist_ok=True)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join("models", "pickplace", "best"),
            log_path=os.path.join("logs", "pickplace_eval"),
            eval_freq=10_000,
            n_eval_episodes=20,
            deterministic=True,
            render=False,
        )
        checkpoint_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=os.path.join("models", "pickplace", "checkpoints"),
            name_prefix="pickplace",
        )
        return CallbackList([eval_cb, checkpoint_cb])

    callbacks = make_pickplace_callbacks(eval_env)

    # ---- Train ----
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        log_interval=10,
    )

    # ---- Save final model ----
    final_path = os.path.join("models", "pickplace", "final")
    os.makedirs(final_path, exist_ok=True)
    model.save(os.path.join(final_path, "tqc_pickplace_final"))
    print(f"\nFinal model saved to {final_path}")

    train_env.close()
    eval_env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
