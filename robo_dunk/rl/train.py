import os

from preprocessing import GrayScaleObservation, ResizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from envs.env import RoboDunkConfig, RoboDunkEnv
from rl.utils import setup_colab


def make_env_fn(env_cfg):
    """
    Return a function that creates an env. This is used by DummyVecEnv.
    We apply GrayScaleObservation and ResizeObservation here so each env
    produces (96,96,1) images (H,W,C) which SB3 VecTransposeImage will then
    convert to (C,H,W).
    """

    def _init():
        config = RoboDunkConfig(
            screen_width=env_cfg.get("screen_width", 400),
            screen_height=env_cfg.get("screen_height", 400),
            fps=env_cfg.get("fps", 60),
            arm_length=env_cfg.get("arm_length", 80),
            bucket_height=env_cfg.get("bucket_height", 10),
            bucket_width=env_cfg.get("bucket_width", 100),
            bucket_y=env_cfg.get("bucket_y", 250),
        )
        # render_mode should be None for training; set "human" if you want visualization
        env = RoboDunkEnv(render_mode=env_cfg.get("render_mode", None), config=config)

        # Convert to grayscale and resize to desired model input (96x96)
        # Keep channel dim so shape is (H, W, 1)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, env_cfg.get("resize", (96, 96)))

        # Monitor for logging episode rewards / lengths
        env = Monitor(env)
        return env

    return _init


def create_vec_env(env_cfg, n_envs=1, frame_stack=4):
    env_fns = [make_env_fn(env_cfg) for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecTransposeImage(vec_env)
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    return vec_env


def create_eval_callback(env_cfg, save_path, n_stack=4, eval_freq=5000):
    eval_env = create_vec_env(env_cfg, n_envs=1, frame_stack=n_stack)
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
    colab_cfg = cfg.get("colab", {})
    env_cfg = cfg.get("env", {})
    ppo_cfg = cfg.get("ppo", {})
    train_cfg = cfg.get("train", {})

    # Policy
    policy = "CnnPolicy"

    # Handle Colab
    ppo_cfg, train_cfg = setup_colab(colab_cfg, ppo_cfg, train_cfg)

    # Tensorboard dir
    if "tensorboard_log" in ppo_cfg and ppo_cfg["tensorboard_log"]:
        os.makedirs(ppo_cfg["tensorboard_log"], exist_ok=True)

    # Create vectorized env
    n_envs = train_cfg.get("n_envs", 1)
    n_stack = train_cfg.get("frame_stack", 4)
    vec_env = create_vec_env(env_cfg, n_envs=n_envs, frame_stack=n_stack)

    # Prepare PPO kwargs
    ppo_kwargs = {k: v for k, v in ppo_cfg.items() if k != "policy"}

    # Create PPO model
    model = PPO(policy, vec_env, **ppo_kwargs)

    # Eval callback
    eval_freq = train_cfg.get("eval_freq", 5000)
    save_path = train_cfg.get("save_path", "./models/ppo_robo_dunk")
    eval_callback = create_eval_callback(
        env_cfg, save_path, n_stack=n_stack, eval_freq=eval_freq
    )

    # Train
    total_timesteps = train_cfg.get("total_timesteps", 200_000)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print("Training completed. Model saved to:", save_path)
    return model
