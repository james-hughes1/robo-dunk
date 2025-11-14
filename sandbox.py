import time
from datetime import datetime

import boto3
import streamlit as st
from stable_baselines3 import PPO

from robo_dunk.envs.factory import InferenceEnv


# CloudWatch client (initialized once)
@st.cache_resource
def get_cloudwatch_client():
    """Initialize CloudWatch client once and reuse across all sessions."""
    return boto3.client("cloudwatch", region_name="eu-west-2")  # Change to your region


cloudwatch = get_cloudwatch_client()


def send_metric(metric_name, value, unit="None"):
    """Send a metric to CloudWatch."""
    try:
        cloudwatch.put_metric_data(
            Namespace="RoboDunk/App",
            MetricData=[
                {
                    "MetricName": metric_name,
                    "Value": value,
                    "Unit": unit,
                    "Timestamp": datetime.now(),
                }
            ],
        )
    except Exception as e:
        print(f"Failed to send metric {metric_name}: {e}")


# --- Streamlit UI ---
st.set_page_config(page_title="RL Sandbox", layout="wide")
st.title("PPO RoboDunk Sandbox")

# Sidebar controls
st.sidebar.header("Environment Controls")
view_style = st.sidebar.selectbox("View Style", ["Original", "Model Input"])
bucket_height = st.sidebar.slider("Lip Height", 10, 50, 20)
bucket_width = st.sidebar.slider("Bucket Width", 70, 100, 80)
bucket_y = st.sidebar.slider("Bucket Height", 200, 300, 250)
arm_length = st.sidebar.slider("Arm Length", 40, 80, 80)
ball_freq = st.sidebar.slider("Ball Frequency", 100, 300, 300)

# Constants
FPS = 60
MAX_STEPS = 1000

# Buttons
start_button = st.sidebar.button("▶️ Start")
pause_button = st.sidebar.button("⏸️ Pause")
reset_button = st.sidebar.button("🔄 Apply")

# Session state
if "running" not in st.session_state:
    st.session_state.running = False
if "inf_env" not in st.session_state:
    st.session_state.inf_env = None
if "obs" not in st.session_state:
    st.session_state.obs = None
if "frame_placeholder" not in st.session_state:
    st.session_state.frame_placeholder = st.empty()

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
raw = view_style == "Original"


# Load model
@st.cache_resource
def load_model(path="./models/ppo_robo_dunk_1919_08112025.zip"):
    return PPO.load(path, device="cpu")


model = load_model()

# Reset env
if reset_button or st.session_state.inf_env is None:
    st.session_state.inf_env = InferenceEnv(
        model, env_cfg, 0.0, render_pygame=False, tracking=True
    )
    st.session_state.running = False

# Start/Stop toggle
if start_button:
    st.session_state.running = True
if pause_button:
    st.session_state.running = False

# Run loop
if st.session_state.running:
    while not st.session_state.inf_env.done:
        # Time the inference
        inference_time = st.session_state.inf_env.step()

        # Send latency metric
        send_metric("InferenceLatency", inference_time * 1000, "Milliseconds")
        frame_image = st.session_state.inf_env.get_obs(max_width=500, raw=raw)
        st.session_state.frame_placeholder.image(
            frame_image, caption="Agent View", width="content"
        )

        if st.session_state.inf_env.done:
            # Send episode metrics
            score, total_reward = st.session_state.inf_env.get_metrics()
            send_metric("EpisodeTotalReward", total_reward, "None")
            send_metric("EpisodeScore", score, "None")
            send_metric("EpisodesCompleted", 1, "Count")

        time.sleep(1.0 / FPS)

else:
    if st.session_state.inf_env:
        frame_image = st.session_state.inf_env.get_obs(max_width=500, raw=raw)
        st.session_state.frame_placeholder.image(
            frame_image, caption="Agent View", width="content"
        )
