import cv2
import gymnasium as gym
import numpy as np


class GrayScaleObservation(gym.ObservationWrapper):
    """Converts RGB observations to grayscale."""

    def __init__(self, env, keep_dim=True):
        super().__init__(env)
        self.keep_dim = keep_dim
        obs_shape = env.observation_space.shape[:2]
        if keep_dim:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(*obs_shape, 1), dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            gray = np.expand_dims(gray, axis=-1)  # (H, W, 1)
        return gray


class ResizeObservation(gym.ObservationWrapper):
    """Resizes observation to a given shape."""

    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape
        # Preserve channel dim if it exists
        channels = (
            env.observation_space.shape[-1]
            if len(env.observation_space.shape) == 3
            else 1
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*shape, channels), dtype=np.uint8
        )

    def observation(self, obs):
        resized = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        if obs.ndim == 3 and resized.ndim == 2:
            # Add back channel dim
            resized = np.expand_dims(resized, axis=-1)
        return resized
