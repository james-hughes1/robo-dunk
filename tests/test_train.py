import multiprocessing
import os
from unittest import mock

import numpy as np
import pytest
from stable_baselines3 import PPO

import robo_dunk.rl.train as train
from robo_dunk.envs.env import RoboDunkEnv


@pytest.fixture
def env_cfg():
    return {
        "screen_width": 200,
        "screen_height": 200,
        "fps": 60,
        "resize": (96, 96),
        "render_mode": None,
    }


@pytest.fixture
def train_cfg(tmp_path):
    return {
        "colab": {"enabled": False},
        "env": {
            "screen_width": 200,
            "screen_height": 200,
            "fps": 60,
            "resize": (96, 96),
            "render_mode": None,
        },
        "ppo": {
            "learning_rate": 0.001,
            "n_steps": 128,
            "batch_size": 16,
            "n_epochs": 1,
            "gamma": 0.99,
            "verbose": 0,
        },
        "train": {
            "total_timesteps": 1,  # just one step for fast test
            "save_path": str(tmp_path / "ppo_test_model.zip"),
            "n_envs": 2,
            "frame_stack": 2,
            "eval_freq": 1,
        },
    }


def test_make_env_fn(env_cfg):
    env_fn = train.make_env_fn(env_cfg)
    env = env_fn()
    assert isinstance(env.unwrapped, RoboDunkEnv)
    obs = env.reset()[0]
    assert obs.shape[:2] == (96, 96)
    assert obs.shape[-1] == 1
    assert env.action_space.shape[0] == 4


def test_create_vec_env(env_cfg):
    vec_env = train.create_vec_env(env_cfg, n_envs=2, frame_stack=2)
    obs = vec_env.reset()
    assert obs.shape[0] == 2  # batch size
    assert obs.shape[1] == 2  # frame_stack dimension
    assert obs.shape[2:] == (96, 96)  # channel, H, W


def test_create_eval_callback(tmp_path, env_cfg):
    save_path = str(tmp_path / "model.zip")
    callback = train.create_eval_callback(env_cfg, save_path, n_stack=2, eval_freq=1)
    assert hasattr(callback, "eval_env")
    # ensure eval_env is a vectorized env
    obs = callback.eval_env.reset()
    assert obs.shape[1:] == (2, 96, 96)  # frame_stack, C,H,W


def test_train_ppo_runs_fast(train_cfg):
    # Patch PPO.learn to avoid actually running RL
    with mock.patch("robo_dunk.rl.train.PPO.learn", return_value=None) as mock_learn:
        model = train.train_ppo(train_cfg)
        # Check learn was called
        assert mock_learn.called
        # Check model is returned
        assert isinstance(model, PPO)
        # Check save file exists (even if mocked, model.save will create zip)
        assert os.path.exists(train_cfg["train"]["save_path"])


def test_train_ppo_env_step(train_cfg):
    # Patch PPO.learn to avoid actual training
    with mock.patch("robo_dunk.rl.train.PPO.learn", return_value=None):
        model = train.train_ppo(train_cfg)
        # Check model can step in the environment
        vec_env = model.get_env()
        obs = vec_env.reset()
        actions = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
        obs, rewards, dones, infos = vec_env.step(actions)
        assert obs.shape[0] == vec_env.num_envs
        assert isinstance(rewards[0], (float, np.floating))
        assert isinstance(dones[0], np.bool_)
        assert isinstance(infos[0], dict)


def test_vec_env_caps_n_envs(monkeypatch):
    env_cfg = {"screen_width": 64, "screen_height": 64}  # minimal env config
    n_envs_requested = multiprocessing.cpu_count() + 5  # intentionally too high
    frame_stack = 1
    base_seed = 0

    # Capture printed warnings
    messages = []
    monkeypatch.setattr("builtins.print", lambda msg: messages.append(msg))

    vec_env = train.create_vec_env(
        env_cfg, n_envs=n_envs_requested, frame_stack=frame_stack, base_seed=base_seed
    )

    # n_envs should be capped at CPU cores
    assert vec_env.num_envs == multiprocessing.cpu_count()
    assert any("n_envs=" in m for m in messages)


def test_env_seed_reproducibility():
    env_cfg = {"screen_width": 64, "screen_height": 64}
    n_envs = 2
    base_seed = 123

    vec_env1 = train.create_vec_env(
        env_cfg, n_envs=n_envs, base_seed=base_seed, use_subproc=False
    )
    vec_env2 = train.create_vec_env(
        env_cfg, n_envs=n_envs, base_seed=base_seed, use_subproc=False
    )

    obs1 = vec_env1.reset()
    obs2 = vec_env2.reset()

    # Observations should match across runs with same base_seed
    assert np.allclose(obs1, obs2)
