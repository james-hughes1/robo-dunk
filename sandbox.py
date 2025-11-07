import time

import numpy as np
import streamlit as st
import torch
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from robo_dunk.envs.env import RoboDunkConfig, RoboDunkEnv
from robo_dunk.rl.preprocessing import GrayScaleObservation, ResizeObservation


# --- Environment Setup ---
def make_env(env_cfg, seed=0, render_mode="rgb_array"):
    config = RoboDunkConfig(
        screen_width=env_cfg["screen_width"],
        screen_height=env_cfg["screen_height"],
        fps=env_cfg["fps"],
        arm_length=env_cfg["arm_length"],
        bucket_height=env_cfg["bucket_height"],
        bucket_width=env_cfg["bucket_width"],
        bucket_y=env_cfg["bucket_y"],
        max_episode_steps=env_cfg["max_episode_steps"],
    )
    env = RoboDunkEnv(render_mode=render_mode, config=config)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, env_cfg["resize"])
    env.reset(seed=seed)
    return env


def create_vec_env(env_cfg, frame_stack=4, seed=0):
    env = DummyVecEnv([lambda: make_env(env_cfg, seed=seed)])
    env = VecTransposeImage(env)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
    return env


# --- Streamlit UI ---
st.set_page_config(page_title="RL Sandbox", layout="wide")
st.title("PPO RoboDunk Sandbox")

# Sidebar controls
st.sidebar.header("Environment Controls")
screen_width = st.sidebar.slider("Screen Width", 200, 800, 400)
screen_height = st.sidebar.slider("Screen Height", 200, 800, 400)
arm_length = st.sidebar.slider("Arm Length", 50, 150, 80)
bucket_y = st.sidebar.slider("Bucket Y Position", 100, 400, 250)
fps = st.sidebar.slider("FPS", 10, 120, 60)

# Buttons
start_button = st.sidebar.button("▶️ Start")
pause_button = st.sidebar.button("⏸️ Pause")
reset_button = st.sidebar.button("🔄 Reset")

# Session state
if "running" not in st.session_state:
    st.session_state.running = False
if "vec_env" not in st.session_state:
    st.session_state.vec_env = None
if "obs" not in st.session_state:
    st.session_state.obs = None
if "pause" not in st.session_state:
    st.session_state.pause = True

# Env config
env_cfg = {
    "screen_width": screen_width,
    "screen_height": screen_height,
    "fps": fps,
    "arm_length": arm_length,
    "bucket_height": 10,
    "bucket_width": 100,
    "bucket_y": bucket_y,
    "max_episode_steps": 1000,
    "resize": (96, 96),
}


# Load model
@st.cache_resource
def load_model(path="./models/ppo_robo_dunk.zip"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return PPO.load(path, device=device)


model = load_model()

# Reset env
if reset_button or st.session_state.vec_env is None:
    st.session_state.vec_env = create_vec_env(env_cfg)
    st.session_state.obs = st.session_state.vec_env.reset()
    st.session_state.running = False

# Start/Stop toggle
if start_button:
    st.session_state.running = True
    st.session_state.pause = False
if pause_button:
    st.session_state.pause = True

# Frame display
frame_placeholder = st.empty()


def step_model_env(model, vec_env, max_width=500):
    # Predict model actions and step environment
    action, _ = model.predict(st.session_state.obs, deterministic=True)
    obs, _, done, _ = st.session_state.vec_env.step(action)
    st.session_state.obs = obs

    # Get frame
    frame = obs[0]
    frame = np.squeeze(obs)[-1]

    # Resize frame
    aspect_ratio = frame.shape[1] / frame.shape[0]
    new_height = int(max_width / aspect_ratio)
    frame_image = Image.fromarray(frame).resize((max_width, new_height))

    return frame_image, done


frame_image, done = step_model_env(model, st.session_state.vec_env, max_width=500)
frame_placeholder.image(frame_image, caption="Agent View", width="content")

# Run loop
if st.session_state.running:
    steps = 0
    while steps < 1000:
        if not st.session_state.pause:
            steps += 1
            frame_image, done = step_model_env(
                model, st.session_state.vec_env, max_width=500
            )
            frame_placeholder.image(frame_image, caption="Agent View", width="content")

        if done[0]:
            st.session_state.obs = st.session_state.vec_env.reset()

        time.sleep(1.0 / fps)
