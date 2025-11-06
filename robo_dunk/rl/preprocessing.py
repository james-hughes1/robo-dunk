import cv2
import gymnasium as gym
import numpy as np


class GrayScaleObservation(gym.ObservationWrapper):
    """
    Converts observations to grayscale.
    If the observation is already grayscale (H,W) or (H,W,1), does nothing.
    """

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
        # Already grayscale
        if obs.ndim == 2:
            gray = obs
            if self.keep_dim:
                gray = np.expand_dims(gray, axis=-1)
            return gray
        elif obs.ndim == 3 and obs.shape[-1] == 1:
            # already single channel
            gray = obs
            if not self.keep_dim:
                gray = gray.squeeze(-1)
            return gray
        elif obs.ndim == 3 and obs.shape[-1] == 3:
            # RGB → grayscale
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            if self.keep_dim:
                gray = np.expand_dims(gray, axis=-1)
            return gray
        else:
            raise ValueError(
                f"Unexpected observation shape {obs.shape} in GrayScaleObservation"
            )


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
