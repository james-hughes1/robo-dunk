import pygame
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from robo_dunk.envs.env import RoboDunkConfig, RoboDunkEnv
from robo_dunk.rl.preprocessing import GrayScaleObservation, ResizeObservation


def make_env(env_cfg, seed=0, render_mode="human"):
    """Create a single RoboDunkEnv with preprocessing for live viewing."""
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

    env = RoboDunkEnv(render_mode=render_mode, config=config)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, env_cfg.get("resize", (96, 96)))
    env.reset(seed=seed)
    return env


def create_view_env(env_cfg, frame_stack=4, seed=0):
    """Create a vectorized, frame-stacked, preprocessed environment for viewing."""
    env = DummyVecEnv([lambda: make_env(env_cfg, seed=seed, render_mode="human")])
    env = VecTransposeImage(env)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
    return env


def view_model_play(model_path, env_cfg, frame_stack=4, max_steps=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create vectorized env for model input (no rendering)
    vec_env = create_view_env(env_cfg, frame_stack=frame_stack)

    # Create raw env for rendering only
    render_env = make_env(env_cfg, seed=0, render_mode="human")

    # Load model
    model = PPO.load(model_path, device=device)

    # Reset both envs
    obs = vec_env.reset()
    render_env.reset()

    clock = pygame.time.Clock()
    step = 0

    while step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = vec_env.step(action)

        # Step the render env with the same action
        render_env.step(action[0])  # unwrap action from batch

        print(action[0])

        clock.tick(env_cfg.get("fps", 60))

        if done[0]:
            obs = vec_env.reset()
            render_env.reset()

        vec_env.render()

        step += 1

    vec_env.close()
    render_env.close()
    print("Playback finished.")


if __name__ == "__main__":
    # Example config — match this to your training config
    env_cfg = {
        "screen_width": 400,
        "screen_height": 400,
        "fps": 60,
        "arm_length": 80,
        "bucket_height": 10,
        "bucket_width": 100,
        "bucket_y": 250,
        "max_episode_steps": 1000,
        "resize": (96, 96),
    }

    model_path = "./models/ppo_robo_dunk.zip"
    view_model_play(model_path, env_cfg, frame_stack=4, max_steps=1000)
