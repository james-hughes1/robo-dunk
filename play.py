import argparse

import numpy as np
import pygame

from robo_dunk.envs.env import RoboDunkConfig, RoboDunkEnv

parser = argparse.ArgumentParser()
parser.add_argument("-s", type=int, default=1000)
parser.add_argument("-d", type=float, default=0.1)

args = parser.parse_args()
max_steps = max(args.s, 10)
difficulty = min(max(args.d, 0.0), 1.0)

play_config = RoboDunkConfig(max_episode_steps=max_steps)

env = RoboDunkEnv(render_mode="human", config=play_config)
env.set_difficulty(difficulty)
obs, _ = env.reset()

running = True
while running:
    # [left, right, arm ac/w, arm c/w]
    action = np.array([0, 0, 0, 0], dtype=np.int8)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action[0] = 1
    if keys[pygame.K_RIGHT]:
        action[1] = 1
    if keys[pygame.K_d]:
        action[2] = 1
    if keys[pygame.K_a]:
        action[3] = 1

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        running = False
    env.render()

env.close()
