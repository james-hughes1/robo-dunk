import time

import streamlit as st
from stable_baselines3 import PPO

from robo_dunk.envs.factory import InferenceEnv

# --- Streamlit UI ---
st.set_page_config(page_title="RL Sandbox", layout="wide")
st.title("PPO RoboDunk Sandbox")

# Sidebar controls
st.sidebar.header("Environment Controls")
bucket_height = st.sidebar.slider("Lip Height", 10, 50, 20)
bucket_width = st.sidebar.slider("Bucket Width", 70, 100, 80)
bucket_y = st.sidebar.slider("Bucket Height", 200, 300, 250)
arm_length = st.sidebar.slider("Arm Length", 40, 80, 80)
ball_freq = st.sidebar.slider("Ball Frequency", 100, 300, 300)

# Constants
FPS = 60
MAX_STEPS = 2000

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
    "fps": FPS,
    "bucket_height_min": bucket_height,
    "bucket_width_min": bucket_width,
    "bucket_y_min": 400 - bucket_y,
    "arm_length_min": arm_length,
    "ball_freq_min": ball_freq,
    "bucket_height_max": bucket_height,
    "bucket_width_max": bucket_width,
    "bucket_y_max": 400 - bucket_y,
    "arm_length_max": arm_length,
    "ball_freq_max": ball_freq,
    "max_episode_steps": MAX_STEPS,
    "resize": (96, 96),
}


# Load model
@st.cache_resource
def load_model(path="./models/ppo_robo_dunk.zip"):
    return PPO.load(path, device="cpu")


model = load_model()

# Reset env
if reset_button or st.session_state.inf_env is None:
    st.session_state.inf_env = InferenceEnv(model, env_cfg, 0.0, render_pygame=False)
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

        time.sleep(1.0 / FPS)

else:
    frame_image = st.session_state.inf_env.get_obs_resized(max_width=500)
    frame_placeholder.image(frame_image, caption="Agent View", width="content")
