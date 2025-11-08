import numpy as np
import pytest

from robo_dunk.envs.env import RoboDunkConfig, RoboDunkEnv


@pytest.fixture
def env():
    config = RoboDunkConfig()
    env = RoboDunkEnv(render_mode=None, config=config)
    yield env
    env.close()


def test_env_initialization(env):
    assert env.screen_width == 400
    assert env.screen_height == 400
    assert env.action_space.shape == (4,)
    assert env.observation_space.shape == (400, 400, 1)


def test_reset_returns_valid_observation(env):
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (400, 400, 1)
    assert env.observation_space.contains(obs)


def test_step_returns_valid_outputs(env):
    obs, _ = env.reset()
    action = np.array([0, 1, 0, 1], dtype=np.int8)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (400, 400, 1)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_robot_movement_bounds(env):
    env.reset()
    env.robot_body.position = (0, env.robot_body.position.y)
    env.step(np.array([0, 0, 0, 0], dtype=np.int8))
    assert env.robot_body.position.x >= env.robot_width // 2

    env.robot_body.position = (env.screen_width, env.robot_body.position.y)
    env.step(np.array([0, 0, 0, 0], dtype=np.int8))
    assert env.robot_body.position.x <= env.screen_width // 2


def test_arm_angle_bounds(env):
    env.reset()
    env.arm_angle = 180
    env.step(np.array([0, 0, 0, 1], dtype=np.int8))
    assert env.arm_angle >= 180

    env.arm_angle = 360
    env.step(np.array([0, 0, 1, 0], dtype=np.int8))
    assert env.arm_angle <= 360


def test_ball_dunk_reward(env):
    env.reset()
    env.ball_body.position = (
        env.screen_width - env.bucket_width // 2,
        env.bucket_y - 1,
    )
    env.ball_body.velocity = (0, 0)
    _, reward, _, _, _ = env.step(np.array([0, 0, 0, 0], dtype=np.int8))
    assert reward == 10 - env.time_penalty


def test_proximity_reward(env):
    env.reset()
    env.ball_body.position = (
        env.screen_width - env.bucket_width // 2 + 10,
        env.bucket_y - 10,
    )
    env.ball_body.velocity = (0, -100)
    _, reward, _, _, _ = env.step(np.array([0, 0, 0, 0], dtype=np.int8))
    assert 0 < reward < 1


def test_robo_dunk_env_episode_termination():
    max_steps = 10
    env = RoboDunkEnv(
        render_mode=None, config=RoboDunkConfig(max_episode_steps=max_steps)
    )

    obs, info = env.reset()

    terminated = False
    for _ in range(max_steps):
        action = env.action_space.sample()
        assert terminated is False
        obs, reward, terminated, truncated, info = env.step(action)

    # After max_steps, the next step should trigger termination
    assert terminated is True

    env.close()
