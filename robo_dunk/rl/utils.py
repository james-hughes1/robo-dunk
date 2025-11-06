import os
import time
from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback


class RolloutProgressCallback(BaseCallback):
    """
    Callback that prints progress during PPO rollout collection.
    Prints every `log_interval` seconds.
    """

    def __init__(self, log_interval=5, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.last_log = time.time()

    def _on_step(self) -> bool:
        now = time.time()
        if now - self.last_log > self.log_interval:
            self.last_log = now
            print(
                f"[Rollout] timestep: {self.num_timesteps}, "
                f"updates: {self.model.num_timesteps // self.model.n_steps}, "
                f"fps: {int(self.model._episode_num/self.model._elapsed_time)}"
            )
        return True


def setup_colab(colab_cfg, ppo_cfg, train_cfg):
    """
    Mount Google Drive if running in Colab and redirect save/log paths
    with timestamped folders.
    """
    if not colab_cfg.get("enabled", False):
        return ppo_cfg, train_cfg

    # Base path for project in Drive
    drive_path = os.path.join(
        colab_cfg.get("drive_mount_point", "/content/drive"),
        colab_cfg.get("drive_project_dir", "MyDrive/robo_dunk_rl"),
    )
    os.makedirs(drive_path, exist_ok=True)

    # Timestamped folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = os.path.join(drive_path, f"run_{timestamp}")
    os.makedirs(run_path, exist_ok=True)

    # Update PPO and train configs
    ppo_cfg["tensorboard_log"] = os.path.join(run_path, "tensorboard/")
    train_cfg["save_path"] = os.path.join(run_path, "models/ppo_robo_dunk")
    print("Colab mode: outputs will be saved to:", run_path)

    return ppo_cfg, train_cfg
