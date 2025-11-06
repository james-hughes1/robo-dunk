import gymnasium as gym
import numpy as np
import pytest

from robo_dunk.envs.env import RoboDunkEnv

# Assuming your custom wrappers are defined in robo_dunk.rl.wrappers
from robo_dunk.rl.preprocessing import GrayScaleObservation, ResizeObservation


@pytest.fixture
def dummy_env():
    env = RoboDunkEnv(render_mode=None)
    # ensure observation is RGB (fake if necessary)
    obs = np.random.randint(
        0,
        256,
        size=(env.config.screen_height, env.config.screen_width, 3),
        dtype=np.uint8,
    )
    env.reset = lambda **kwargs: (obs, {})  # monkey-patch reset for testing
    return env


def test_grayscale_observation_shape(dummy_env):
    env = GrayScaleObservation(dummy_env, keep_dim=True)
    obs, _ = env.reset()
    assert obs.ndim == 3
    assert obs.shape[-1] == 1
    # Pixel values should be in 0-255
    assert obs.min() >= 0 and obs.max() <= 255


def test_resize_observation_shape(dummy_env):
    target_shape = (96, 96)
    env = ResizeObservation(dummy_env, target_shape)
    obs, _ = env.reset()
    assert obs.shape[:2] == target_shape
    # Check pixel values are preserved
    assert obs.min() >= 0 and obs.max() <= 255


def test_stack_grayscale_and_resize(dummy_env):
    target_shape = (64, 64)
    env = GrayScaleObservation(dummy_env, keep_dim=True)
    env = ResizeObservation(env, target_shape)
    obs, _ = env.reset()
    assert obs.shape == (*target_shape, 1)
    assert obs.dtype == np.uint8


def test_grayscale_without_keep_dim(dummy_env):
    env = GrayScaleObservation(dummy_env, keep_dim=False)
    obs, _ = env.reset()
    assert obs.ndim == 2  # no channel dimension
    assert obs.dtype == np.uint8


def test_resize_with_single_channel():
    # Create a dummy single-channel env
    class DummySingleChannelEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(50, 50, 1), dtype=np.uint8
            )
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, **kwargs):
            return np.random.randint(0, 256, (50, 50, 1), dtype=np.uint8), {}

        def step(self, action):
            return (
                np.random.randint(0, 256, (50, 50, 1), dtype=np.uint8),
                0.0,
                False,
                False,
                {},
            )

    env = ResizeObservation(DummySingleChannelEnv(), (32, 32))
    obs, _ = env.reset()
    assert obs.shape == (32, 32, 1)
