import os
import time
from datetime import datetime
from pathlib import Path

import boto3
import gradio as gr
from stable_baselines3 import PPO

from robo_dunk.envs.factory import InferenceEnv

# Get models directory from environment variable
MODELS_DIR = os.getenv("MODELS_DIR", "./models")

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
        self.inf_env = None
        self.model = None
        self.model_name = None
        self.running = False
        self.inference_times = []
        self.thread = None


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
        state.inf_env = None
        return f"✅ Model loaded: {model_name}"
    return f"Model already loaded: {model_name}"


def reset_env(
    model_name, view_style, bucket_height, bucket_width, bucket_y, arm_length, ball_freq
):
    """Reset environment with current parameters."""
    state.running = False
    if state.model is None:
        load_model_action(model_name)

    state.inf_env = create_env(
        state.model, bucket_height, bucket_width, bucket_y, arm_length, ball_freq
    )
    state.inference_times = []

    raw = view_style == "Original"
    frame = state.inf_env.get_obs(max_width=400, raw=raw)
    return frame, "Environment reset"


def start_episode(
    model_name, view_style, bucket_height, bucket_width, bucket_y, arm_length, ball_freq
):
    """Start or resume episode."""
    if state.model is None:
        load_model_action(model_name)

    if state.inf_env is None or state.inf_env.done:
        state.inf_env = create_env(
            state.model, bucket_height, bucket_width, bucket_y, arm_length, ball_freq
        )
        state.inference_times = []

    state.running = True
    raw = view_style == "Original"

    # Run episode in generator mode for smooth updates
    while state.running and not state.inf_env.done:
        inference_time = state.inf_env.step()
        state.inference_times.append(inference_time)

        frame = state.inf_env.get_obs(max_width=400, raw=raw)

        if state.inf_env.done:
            avg_inference_time = sum(state.inference_times) / len(state.inference_times)
            score, total_reward = state.inf_env.get_metrics()
            send_metrics(avg_inference_time, total_reward, score)
            state.running = False
            yield (
                frame,
                f"✅ Episode Complete! Score: {score}," + f"Reward: {total_reward:.2f}",
            )
        else:
            yield frame, f"Running... (Frame {len(state.inference_times)})"

        time.sleep(1.0 / 30)


def pause_episode():
    """Pause the episode."""
    state.running = False
    return "Paused"


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
            ball_freq = gr.Slider(100, 300, value=300, label="Ball Frequency")

            gr.Markdown("## Controls")
            with gr.Row():
                play_btn = gr.Button("▶️ Play", variant="primary")
                pause_btn = gr.Button("⏸️ Pause")
            reset_btn = gr.Button("🔄 Apply / Reset")

        with gr.Column(scale=2):
            video_output = gr.Image(label="Agent View", height=400)
            episode_status = gr.Textbox(label="Episode Status", interactive=False)

    # Wire up callbacks
    if available_models:
        load_btn.click(
            fn=load_model_action, inputs=[model_dropdown], outputs=[model_status]
        )

        reset_btn.click(
            fn=reset_env,
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

        play_btn.click(
            fn=start_episode,
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

        pause_btn.click(fn=pause_episode, outputs=[episode_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8501)
