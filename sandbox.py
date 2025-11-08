import time

import streamlit as st
from stable_baselines3 import PPO

from robo_dunk.envs.factory import InferenceEnv

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
if "inf_env" not in st.session_state:
    st.session_state.inf_env = None
if "obs" not in st.session_state:
    st.session_state.obs = None

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
    return PPO.load(path, device="cpu")


model = load_model()

# Reset env
if reset_button or st.session_state.inf_env is None:
    st.session_state.inf_env = InferenceEnv(model, env_cfg, render_pygame=False)
    st.session_state.running = False

# Start/Stop toggle
if start_button:
    st.session_state.running = True
if pause_button:
    st.session_state.running = False

# Frame display
frame_placeholder = st.empty()

# Run loop
if st.session_state.running:
    while not st.session_state.inf_env.done:
        st.session_state.inf_env.step()
        frame_image = st.session_state.inf_env.get_obs_resized(max_width=500)
        frame_placeholder.image(frame_image, caption="Agent View", width="content")

        if st.session_state.inf_env.done:
            st.session_state.inf_env.reset()

        time.sleep(1.0 / fps)

else:
    frame_image = st.session_state.inf_env.get_obs_resized(max_width=500)
    frame_placeholder.image(frame_image, caption="Agent View", width="content")
