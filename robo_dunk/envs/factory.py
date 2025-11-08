import multiprocessing

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from robo_dunk.envs.env import RoboDunkConfig, RoboDunkEnv
from robo_dunk.rl.preprocessing import GrayScaleObservation, ResizeObservation


def make_env(
    env_cfg,
    render_mode,
    seed=0,
    curriculum_wr=None,
    monitor_fn=None,
    difficulty=0.0,
    deterministic=False,
):
    """Create a single RoboDunkEnv with preprocessing for live viewing."""
    config = RoboDunkConfig(
        fps=env_cfg.get("fps", 60),
        max_episode_steps=env_cfg.get("max_episode_steps", 1000),
        bucket_height_min=env_cfg.get("bucket_height_min", 10),
        bucket_height_max=env_cfg.get("bucket_height_max", 50),
        bucket_width_min=env_cfg.get("bucket_width_min", 70),
        bucket_width_max=env_cfg.get("bucket_width_max", 100),
        bucket_y_min=env_cfg.get("bucket_y_min", 100),
        bucket_y_max=env_cfg.get("bucket_y_max", 200),
        arm_length_min=env_cfg.get("arm_length_min", 40),
        arm_length_max=env_cfg.get("arm_length_max", 80),
        ball_freq_min=env_cfg.get("ball_freq_min", 200),
        ball_freq_max=env_cfg.get("ball_freq_max", 300),
        proximity_reward=env_cfg.get("proximity_reward", 10.0),
        time_penalty=env_cfg.get("time_penalty", 0.01),
    )

    env = RoboDunkEnv(render_mode=render_mode, config=config)
    env.set_difficulty(difficulty_level=difficulty, deterministic=deterministic)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, env_cfg.get("resize", (96, 96)))

    if curriculum_wr:

        def difficulty_schedule(step):
            return step / config.max_episode_steps

        env = curriculum_wr(env, difficulty_schedule)

    if monitor_fn:
        env = monitor_fn(env)

    env.reset(seed=seed)
    return env


def make_env_fn(
    env_cfg,
    rank=0,
    base_seed=0,
    curriculum_wr=None,
    monitor_fn=None,
    difficulty=0.0,
    deterministic=False,
):
    """
    Return a function that creates an env. Each env gets a unique seed
    for reproducibility across runs.
    """

    def _init():
        seed = base_seed + rank
        env = make_env(
            env_cfg,
            render_mode="rgb_array",
            seed=seed,
            curriculum_wr=curriculum_wr,
            monitor_fn=monitor_fn,
            difficulty=difficulty,
            deterministic=deterministic,
        )
        return env

    return _init


def create_vec_env(
    env_cfg,
    n_envs=1,
    frame_stack=4,
    base_seed=0,
    use_subproc=True,
    curriculum_wr=None,
    monitor_fn=None,
    difficulty=0.0,
):
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

    env_fns = [
        make_env_fn(
            env_cfg,
            rank=i,
            base_seed=base_seed,
            curriculum_wr=curriculum_wr,
            monitor_fn=monitor_fn,
            difficulty=difficulty,
        )
        for i in range(n_envs)
    ]

    vec_env_cls = SubprocVecEnv if use_subproc and n_envs > 1 else DummyVecEnv
    vec_env = vec_env_cls(env_fns)
    vec_env = VecTransposeImage(vec_env)
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    return vec_env


def create_view_env(env_cfg, render_mode, frame_stack=4, seed=0, difficulty=0.0):
    """Create a vectorized, frame-stacked, preprocessed environment for viewing."""
    env = DummyVecEnv(
        [
            lambda: make_env(
                env_cfg,
                seed=seed,
                render_mode=render_mode,
                difficulty=difficulty,
                deterministic=True,
            )
        ]
    )
    env = VecTransposeImage(env)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
    return env


class InferenceEnv:
    def __init__(self, model, env_cfg, difficulty, render_pygame=False):
        self.env = create_view_env(
            env_cfg,
            render_mode=("human" if render_pygame else "rgb_array"),
            frame_stack=4,
            seed=0,
            difficulty=difficulty,
        )
        self.model = model
        self.render_pygame = render_pygame

        self.steps = 0
        self.obs = self.env.reset()
        self.frame = self._obs_to_frame(self.obs)
        self.done = False

    def step(self):
        action, _ = self.model.predict(self.obs, deterministic=True)
        obs, _, done, _ = self.env.step(action)
        self.obs = obs

        self.steps += 1

        self.frame = self._obs_to_frame(self.obs)

        if self.render_pygame:
            self.env.render()

        self.done = done[0]

    def _obs_to_frame(self, obs):
        frame = obs[0]
        frame = np.squeeze(obs)[-1]
        return frame

    def get_obs(self, max_width, raw=False):
        if raw:
            frame = self.env.envs[0].env.env.get_obs_raw(decorations=True)
        else:
            frame = self.frame
        # Resize frame
        aspect_ratio = frame.shape[1] / frame.shape[0]
        new_height = int(max_width / aspect_ratio)
        frame_resized = Image.fromarray(frame).resize((max_width, new_height))
        return frame_resized

    def reset(self):
        self.steps = 0
        self.obs = self.env.reset()
        self.frame = self._obs_to_frame(self.obs)
        self.done = False

    def close(self):
        self.env.close()


class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, difficulty_schedule):
        super().__init__(env)
        self.difficulty_schedule = difficulty_schedule
        self.global_step = 0

    def reset(self, **kwargs):
        difficulty = self.difficulty_schedule(self.global_step)
        if hasattr(self.env, "set_difficulty"):
            self.env.set_difficulty(difficulty)
        return self.env.reset(**kwargs)

    def step(self, action):
        self.global_step += 1
        return self.env.step(action)
