import argparse

import pygame
from stable_baselines3 import PPO

from robo_dunk.envs.factory import InferenceEnv

parser = argparse.ArgumentParser()
parser.add_argument("-s", type=int, default=1000)
parser.add_argument("-d", type=float, default=0.1)

args = parser.parse_args()
max_steps = max(args.s, 10)
difficulty = min(max(args.d, 0.0), 1.0)

if __name__ == "__main__":
    # Example config — match this to your training config
    env_cfg = {}

    model_path = "./models/ppo_robo_dunk.zip"
    model = PPO.load(model_path, device="cpu")

    inf_env = InferenceEnv(model, env_cfg, difficulty=difficulty, render_pygame=True)

    clock = pygame.time.Clock()

    while not inf_env.done:
        inf_env.step()
        clock.tick(env_cfg.get("fps", 60))

    inf_env.close()
    print("Playback finished.")
