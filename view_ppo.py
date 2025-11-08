import pygame
from stable_baselines3 import PPO

from robo_dunk.envs.factory import InferenceEnv

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
    model = PPO.load(model_path, device="cpu")

    inf_env = InferenceEnv(model, env_cfg, render_pygame=True)

    clock = pygame.time.Clock()

    while not inf_env.done:
        inf_env.step()
        clock.tick(env_cfg.get("fps", 60))

    inf_env.close()
    print("Playback finished.")
