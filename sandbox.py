import os
from datetime import datetime
from pathlib import Path

import boto3
import gradio as gr
import imageio
import numpy as np
from stable_baselines3 import PPO

from robo_dunk.envs.factory import InferenceEnv

# Get models directory from environment variable
MODELS_DIR = os.getenv("MODELS_DIR", "./models")

# Create videos directory
VIDEOS_DIR = "./videos"
os.makedirs(VIDEOS_DIR, exist_ok=True)

# CloudWatch client
cloudwatch = boto3.client("cloudwatch", region_name="eu-west-2")


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
    model_files = sorted([f.name for f in models_path.glob("*.zip")])
    return model_files


def load_model(model_name):
    """Load a specific model by name."""
    model_path = f"{MODELS_DIR}/{model_name}"
    print(f"Loading model: {model_path}")
    return PPO.load(model_path, device="cpu")


# Global state
class AppState:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.current_video_path = None


state = AppState()


def create_env(model, bucket_height, bucket_width, bucket_y, arm_length, ball_freq):
    """Create a new environment with given parameters."""
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
        "max_episode_steps": 1000,
        "resize": (96, 96),
    }
    return InferenceEnv(model, env_cfg, 0.0, render_pygame=False, tracking=True)


def load_model_action(model_name):
    """Load selected model."""
    if model_name != state.model_name:
        state.model = load_model(model_name)
        state.model_name = model_name
        return f"✅ Model loaded: {model_name}"
    return f"Model already loaded: {model_name}"


def compute_episode(
    model_name,
    view_style,
    bucket_height,
    bucket_width,
    bucket_y,
    arm_length,
    ball_freq,
    progress=gr.Progress(),
):
    """Compute entire episode and save as MP4 video."""
    if state.model is None:
        load_model_action(model_name)

    # Create environment
    inf_env = create_env(
        state.model, bucket_height, bucket_width, bucket_y, arm_length, ball_freq
    )
    raw = view_style == "Original"

    # Create video file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(VIDEOS_DIR, f"episode_{timestamp}.mp4")

    # Collect all frames first
    frames = []
    inference_times = []

    progress(0, desc="Computing episode...")

    # Get first frame
    first_frame = inf_env.get_obs(max_width=400, raw=raw)
    frames.append(np.array(first_frame))

    frame_count = 1

    while not inf_env.done:
        inference_time = inf_env.step()
        inference_times.append(inference_time)

        frame = inf_env.get_obs(max_width=400, raw=raw)
        frames.append(np.array(frame))

        frame_count += 1
        progress(
            min(frame_count / 1000, 0.99), desc=f"Computing frame {frame_count}/1000..."
        )

    # Write video using imageio (has built-in ffmpeg)
    progress(0.99, desc="Writing video file...")
    imageio.mimwrite(video_path, frames, fps=60, codec="libx264", quality=8)

    # Calculate metrics
    avg_inference_time = sum(inference_times) / len(inference_times)
    score, total_reward = inf_env.get_metrics()

    # Send to CloudWatch
    send_metrics(avg_inference_time, total_reward, score)

    # Clean up
    inf_env.close()

    # Store video path
    state.current_video_path = video_path

    progress(1.0, desc="Complete!")

    print(f"Video saved to: {video_path}")
    print(f"Video file exists: {os.path.exists(video_path)}")
    print(f"Video file size: {os.path.getsize(video_path)} bytes")

    message = (
        "✅ Episode Complete!\n\n"
        f"Score: {score}\n"
        f"Total Reward: {total_reward:.2f}\n"
        f"Frames: {frame_count}\n"
        f"Avg Inference: {avg_inference_time * 1000:.2f}ms"
    )

    return video_path, message


# Build Gradio interface
with gr.Blocks(title="RL Sandbox") as demo:
    gr.Markdown("# PPO RoboDunk Sandbox")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Model Selection")

            available_models = get_available_models()
            if not available_models:
                gr.Markdown(f"⚠️ No models found in `{MODELS_DIR}/`")
            else:
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="Choose Model",
                )
                load_btn = gr.Button("Load Model")
                model_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("## Environment Controls")
            view_style = gr.Radio(
                ["Original", "Model Input"], value="Original", label="View Style"
            )
            bucket_height = gr.Slider(10, 50, value=20, label="Lip Height")
            bucket_width = gr.Slider(70, 100, value=80, label="Bucket Width")
            bucket_y = gr.Slider(200, 300, value=250, label="Bucket Height")
            arm_length = gr.Slider(40, 80, value=80, label="Arm Length")
            ball_freq = gr.Slider(100, 300, value=300, label="Ball Reload")

            gr.Markdown("## Controls")
            compute_btn = gr.Button("🎬 Compute Episode", variant="primary", size="lg")

        with gr.Column(scale=2):
            video_output = gr.Video(
                label="Episode Playback", autoplay=False, height=600
            )
            episode_status = gr.Textbox(
                label="Episode Stats", interactive=False, lines=6
            )

    gr.Markdown(
        """
    ### How to use:
    1. Select a model and click "Load Model"
    2. Adjust environment parameters as desired
    3. Click "Compute Episode" to run the agent (this may take 20-30 seconds)
    4. Once complete, the video will play automatically with full playback controls

    The video plays at 60 FPS for smooth playback.
    Use the video controls to pause, seek, or replay!
    """
    )

    # Wire up callbacks
    if available_models:
        load_btn.click(
            fn=load_model_action, inputs=[model_dropdown], outputs=[model_status]
        )

        compute_btn.click(
            fn=compute_episode,
            inputs=[
                model_dropdown,
                view_style,
                bucket_height,
                bucket_width,
                bucket_y,
                arm_length,
                ball_freq,
            ],
            outputs=[video_output, episode_status],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8501)
