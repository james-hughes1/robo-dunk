import os
import time
from datetime import datetime
from pathlib import Path

import boto3
import streamlit as st
from stable_baselines3 import PPO

from robo_dunk.envs.factory import InferenceEnv

# Get models directory from environment variable
MODELS_DIR = os.getenv("MODELS_DIR", "./models")


# CloudWatch client (initialized once)
@st.cache_resource
def get_cloudwatch_client():
    """Initialize CloudWatch client once and reuse across all sessions."""
    return boto3.client("cloudwatch", region_name="eu-west-2")


cloudwatch = get_cloudwatch_client()


def send_metrics(avg_inference_time, total_reward, score):
    """Send all episode metrics to CloudWatch at once."""
    try:
        cloudwatch.put_metric_data(
            Namespace="RoboDunk/App",
            MetricData=[
                {
                    "MetricName": "AvgInferenceLatency",
                    "Value": avg_inference_time * 1000,
                    "Unit": "Milliseconds",
                    "Timestamp": datetime.now(),
                },
                {
                    "MetricName": "EpisodeTotalReward",
                    "Value": total_reward,
                    "Unit": "None",
                    "Timestamp": datetime.now(),
                },
                {
                    "MetricName": "EpisodeScore",
                    "Value": score,
                    "Unit": "None",
                    "Timestamp": datetime.now(),
                },
                {
                    "MetricName": "EpisodesCompleted",
                    "Value": 1,
                    "Unit": "Count",
                    "Timestamp": datetime.now(),
                },
            ],
        )
    except Exception as e:
        print(f"Failed to send metrics: {e}")


def get_available_models(models_dir=MODELS_DIR):
    """Scan the models directory and return list of .zip files."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []

    # Find all .zip files in the models directory
    model_files = sorted([f.name for f in models_path.glob("*.zip")])
    return model_files


# Load model with caching
@st.cache_resource
def load_model(model_name):
    """Load a specific model by name."""
    model_path = f"{MODELS_DIR}/{model_name}"
    print(f"Loading model: {model_path}")
    return PPO.load(model_path, device="cpu")


# --- Streamlit UI ---
st.set_page_config(page_title="RL Sandbox", layout="wide")
st.title("PPO RoboDunk Sandbox")

# Sidebar controls
st.sidebar.header("Model Selection")

# Get available models
available_models = get_available_models()

if not available_models:
    st.sidebar.error(f"No models found in {MODELS_DIR}/")
    st.sidebar.info(f"Please mount models directory with: -v ~/models:{MODELS_DIR}")
    st.stop()

# Model selector
selected_model = st.sidebar.selectbox(
    "Choose Model",
    available_models,
    help="Select a trained model from the mounted models directory",
)

# Display model info
st.sidebar.caption(f"Selected: `{selected_model}`")

st.sidebar.header("Environment Controls")
view_style = st.sidebar.selectbox("View Style", ["Original", "Model Input"])
bucket_height = st.sidebar.slider("Lip Height", 10, 50, 20)
bucket_width = st.sidebar.slider("Bucket Width", 70, 100, 80)
bucket_y = st.sidebar.slider("Bucket Height", 200, 300, 250)
arm_length = st.sidebar.slider("Arm Length", 40, 80, 80)
ball_freq = st.sidebar.slider("Ball Frequency", 100, 300, 300)

# Constants
FPS = 20
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
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "inference_times" not in st.session_state:
    st.session_state.inference_times = []

# Env config
env_cfg = {
    "fps": 60,
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

# Load model (only when changed or first time)
if st.session_state.current_model != selected_model:
    with st.spinner(f"Loading model: {selected_model}..."):
        model = load_model(selected_model)
        st.session_state.current_model = selected_model
        st.session_state.inf_env = None  # Force env reset with new model
        st.success(f"✅ Model loaded: {selected_model}")
else:
    model = load_model(selected_model)

# Reset env
if reset_button or st.session_state.inf_env is None:
    st.session_state.inf_env = InferenceEnv(
        model, env_cfg, 0.0, render_pygame=False, tracking=True
    )
    st.session_state.running = False
    st.session_state.inference_times = []

# Start/Stop toggle
if start_button:
    st.session_state.running = True
    st.session_state.inference_times = []
if pause_button:
    st.session_state.running = False

# Run loop
if st.session_state.running:
    while not st.session_state.inf_env.done:
        # Time the inference
        inference_time = st.session_state.inf_env.step()
        st.session_state.inference_times.append(inference_time)

        frame_image = st.session_state.inf_env.get_obs(max_width=400, raw=raw)
        st.session_state.frame_placeholder.image(
            frame_image, caption="Agent View", width="content"
        )

        if st.session_state.inf_env.done:
            # Calculate average inference time
            avg_inference_time = sum(st.session_state.inference_times) / len(
                st.session_state.inference_times
            )

            # Send all metrics at once
            score, total_reward = st.session_state.inf_env.get_metrics()
            send_metrics(avg_inference_time, total_reward, score)

        time.sleep(1.0 / FPS)

else:
    if st.session_state.inf_env:
        frame_image = st.session_state.inf_env.get_obs(max_width=400, raw=raw)
        st.session_state.frame_placeholder.image(
            frame_image, caption="Agent View", width="content"
        )
