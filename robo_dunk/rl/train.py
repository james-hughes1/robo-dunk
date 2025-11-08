import os

import imageio
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from robo_dunk.envs.factory import InferenceEnv, create_vec_env
from robo_dunk.rl.utils import setup_colab


def record_gif(model, env_cfg, gif_path="play.gif", max_frames=1000):
    """
    Record a GIF using the VecEnv and the original RoboDunkEnv.
    Works with wrappers like GrayScaleObservation and ResizeObservation.
    """
    inf_env = InferenceEnv(model, env_cfg, render_pygame=False)
    frames = []

    for _ in range(max_frames):
        # Step environment
        inf_env.step()
        frames.append(inf_env.frame)

        if inf_env.done:
            break

    inf_env.close()

    if len(frames) > 0:
        imageio.mimsave(gif_path, frames, fps=60)
        print(f"Saved evaluation GIF to: {gif_path}")
    else:
        print("No frames captured; GIF not saved.")


class GifEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env,
        env_cfg,
        gif_path=None,
        **kwargs,
    ):
        super().__init__(eval_env=eval_env, **kwargs)
        self.gif_path = gif_path
        self.env_cfg = env_cfg

    def _on_step(self):
        result = super()._on_step()
        if self.n_calls % self.eval_freq == 0:
            if self.gif_path:
                path = f"{self.gif_path}/eval_{self.n_calls}.gif"
                record_gif(
                    self.model,
                    env_cfg=self.env_cfg,
                    gif_path=path,
                    max_frames=1000,
                )
        return result


def train_ppo(cfg):
    """
    Train PPO on RoboDunk with configurable parallel environments.
    """
    colab_cfg = cfg.get("colab", {})
    env_cfg = cfg.get("env", {})
    ppo_cfg = cfg.get("ppo", {})
    train_cfg = cfg.get("train", {})

    # Handle Colab
    ppo_cfg, train_cfg = setup_colab(colab_cfg, ppo_cfg, train_cfg)

    # Tensorboard directory
    if "tensorboard_log" in ppo_cfg and ppo_cfg["tensorboard_log"]:
        os.makedirs(ppo_cfg["tensorboard_log"], exist_ok=True)

    # Create vectorized envs
    n_envs = train_cfg.get("n_envs", 1)
    n_stack = train_cfg.get("frame_stack", 4)
    base_seed = train_cfg.get("seed", 0)
    vec_env = create_vec_env(
        env_cfg,
        n_envs=n_envs,
        frame_stack=n_stack,
        base_seed=base_seed,
        use_subproc=True,
    )

    # PPO model
    policy = "CnnPolicy"
    ppo_kwargs = {k: v for k, v in ppo_cfg.items() if k != "policy"}

    # Force GPU usage if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = PPO(policy, vec_env, device=device, **ppo_kwargs)

    # Eval callback
    save_path = train_cfg.get("save_path", "./models/ppo_robo_dunk")
    eval_freq = train_cfg.get("eval_freq", 5000)
    eval_vec_env = create_vec_env(
        env_cfg, n_envs=1, frame_stack=n_stack, base_seed=base_seed, use_subproc=False
    )
    eval_callback = GifEvalCallback(
        eval_env=eval_vec_env,
        env_cfg=env_cfg,
        gif_path=os.path.dirname(save_path),
        eval_freq=eval_freq,
        best_model_save_path=os.path.dirname(save_path),
        log_path=os.path.dirname(save_path),
        deterministic=True,
        render=False,
    )

    # Log callback
    logging_freq = train_cfg.get("logging_freq", 10)

    # Train
    total_timesteps = train_cfg.get("total_timesteps", 200_000)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback],
        log_interval=logging_freq,
    )

    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print("Training completed. Model saved to:", save_path)

    return model
