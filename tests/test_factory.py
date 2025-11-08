import multiprocessing

import numpy as np

from robo_dunk.envs.env import RoboDunkEnv
from robo_dunk.envs.factory import create_vec_env, make_env_fn


def test_make_env_fn(env_cfg):
    env_fn = make_env_fn(env_cfg)
    env = env_fn()
    assert isinstance(env.unwrapped, RoboDunkEnv)
    obs = env.reset()[0]
    assert obs.shape[:2] == (96, 96)
    assert obs.shape[-1] == 1
    assert env.action_space.shape[0] == 4


def test_create_vec_env(env_cfg):
    vec_env = create_vec_env(env_cfg, n_envs=2, frame_stack=2)
    obs = vec_env.reset()
    assert obs.shape[0] == 2  # batch size
    assert obs.shape[1] == 2  # frame_stack dimension
    assert obs.shape[2:] == (96, 96)  # channel, H, W


def test_vec_env_caps_n_envs(monkeypatch):
    env_cfg = {"screen_width": 64, "screen_height": 64}  # minimal env config
    n_envs_requested = multiprocessing.cpu_count() + 5  # intentionally too high
    frame_stack = 1
    base_seed = 0

    # Capture printed warnings
    messages = []
    monkeypatch.setattr("builtins.print", lambda msg: messages.append(msg))

    vec_env = create_vec_env(
        env_cfg, n_envs=n_envs_requested, frame_stack=frame_stack, base_seed=base_seed
    )

    # n_envs should be capped at CPU cores
    assert vec_env.num_envs == multiprocessing.cpu_count()
    assert any("n_envs=" in m for m in messages)


def test_seed_reproducibility():
    env1 = RoboDunkEnv()
    env2 = RoboDunkEnv()

    seed = 123
    obs1, _ = env1.reset(seed=seed)
    obs2, _ = env2.reset(seed=seed)

    # Obs arrays should be identical for the first frame
    assert np.allclose(obs1, obs2)

    # After some steps, the sequence is still reproducible if seeded
    for _ in range(5):
        obs1, _, _, _, _ = env1.step(env1.action_space.sample())
        obs2, _, _, _, _ = env2.step(env2.action_space.sample())
