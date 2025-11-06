import multiprocessing
import os

import imageio
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from robo_dunk.envs.env import RoboDunkConfig, RoboDunkEnv
from robo_dunk.rl.preprocessing import GrayScaleObservation, ResizeObservation
from robo_dunk.rl.utils import setup_colab


def make_env_fn(env_cfg, rank=0, base_seed=0):
    """
    Return a function that creates an env. Each env gets a unique seed
    for reproducibility across runs.
    """

    def _init():
        seed = base_seed + rank

        config = RoboDunkConfig(
            screen_width=env_cfg.get("screen_width", 400),
            screen_height=env_cfg.get("screen_height", 400),
            fps=env_cfg.get("fps", 60),
            arm_length=env_cfg.get("arm_length", 80),
            bucket_height=env_cfg.get("bucket_height", 10),
            bucket_width=env_cfg.get("bucket_width", 100),
            bucket_y=env_cfg.get("bucket_y", 250),
            max_episode_steps=env_cfg.get("max_episode_steps", 1000),
        )

        # render_mode=None for training
        env = RoboDunkEnv(render_mode=env_cfg.get("render_mode", None), config=config)

        # Apply custom grayscale + resize wrappers
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, env_cfg.get("resize", (96, 96)))

        # Monitor logs & set seed for reproducibility
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def create_vec_env(env_cfg, n_envs=1, frame_stack=4, base_seed=0, use_subproc=True):
    """
    Create a vectorized environment with optional SubprocVecEnv.
    Caps n_envs to available CPU cores if using SubprocVecEnv.
    """
    max_cores = multiprocessing.cpu_count()
    if use_subproc and n_envs > max_cores:
        print(
            f"n_envs={n_envs} exceeds CPU cores={max_cores}. "
            f"Capping to {max_cores} for efficiency."
        )
        n_envs = max_cores

    env_fns = [make_env_fn(env_cfg, rank=i, base_seed=base_seed) for i in range(n_envs)]
    vec_env_cls = SubprocVecEnv if use_subproc and n_envs > 1 else DummyVecEnv
    vec_env = vec_env_cls(env_fns)
    vec_env = VecTransposeImage(vec_env)
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    return vec_env


def record_gif(model, env_cfg, gif_path="play.gif", max_steps=1000):
    """
    Run a single evaluation episode and save it as a GIF.
    """
    # Create an environment with render_mode='rgb_array' so we can grab frames
    env_cfg = dict(env_cfg)  # copy so we don't modify the original
    env_cfg["render_mode"] = "rgb_array"

    env = create_vec_env(env_cfg, n_envs=1, frame_stack=1, use_subproc=False)
    obs = env.reset()
    frames = []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        # Capture the rendered frame
        frame = env.render(mode="rgb_array")[0]  # VecEnv returns a batch
        frames.append(frame)

        if dones[0]:
            break

    env.close()

    # Save as GIF
    imageio.mimsave(gif_path, frames, fps=env_cfg.get("fps", 30))
    print(f"Saved evaluation GIF to: {gif_path}")


class GifEvalCallback(EvalCallback):
    def __init__(self, *args, gif_path=None, env_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gif_path = gif_path
        self.env_cfg = env_cfg

    def _on_step(self):
        result = super()._on_step()
        if self.n_calls % self.eval_freq == 0:
            if self.gif_path:
                path = f"{self.gif_path}/eval_{self.n_calls}.gif"
                record_gif(self.model, self.env_cfg, path)
        return result


def create_eval_callback(env_cfg, save_path, n_stack=4, eval_freq=5000, base_seed=0):
    """
    Create a single-env evaluation callback.
    """
    eval_env = create_vec_env(
        env_cfg, n_envs=1, frame_stack=n_stack, base_seed=base_seed, use_subproc=False
    )
    best_model_dir = os.path.dirname(save_path)
    os.makedirs(best_model_dir, exist_ok=True)
    return EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=best_model_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )


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
    eval_callback = GifEvalCallback(
        eval_env=create_vec_env(
            env_cfg,
            n_envs=1,
            frame_stack=n_stack,
            base_seed=base_seed,
            use_subproc=False,
        ),
        best_model_save_path=os.path.dirname(save_path),
        log_path=os.path.dirname(save_path),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        gif_path=os.path.dirname(save_path),
        env_cfg=env_cfg,
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
