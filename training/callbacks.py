"""Callback factories for NeuroGenesis training runs."""

import os
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)


def make_reach_callbacks(
    eval_env,
    log_dir: str = "logs",
    model_dir: str = "models",
    eval_freq: int = 10_000,
    checkpoint_freq: int = 50_000,
    n_eval_episodes: int = 20,
) -> CallbackList:
    """Build the standard callback stack for a reach training run.

    Args:
        eval_env: A gymnasium env used for periodic evaluation.
        log_dir: Root directory for TensorBoard logs.
        model_dir: Root directory for saved checkpoints.
        eval_freq: Evaluate every N timesteps.
        checkpoint_freq: Save a checkpoint every N timesteps.
        n_eval_episodes: Episodes per evaluation round.

    Returns:
        A CallbackList ready to pass to ``model.learn()``.
    """
    os.makedirs(os.path.join(model_dir, "reach"), exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "reach", "best"),
        log_path=os.path.join(log_dir, "reach_eval"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(model_dir, "reach", "checkpoints"),
        name_prefix="reach",
    )

    return CallbackList([eval_cb, checkpoint_cb])
