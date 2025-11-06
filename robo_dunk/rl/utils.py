import os
from datetime import datetime


def setup_colab(colab_cfg, ppo_cfg, train_cfg):
    """
    Mount Google Drive if running in Colab and redirect save/log paths
    with timestamped folders.
    """
    if not colab_cfg.get("enabled", False):
        return ppo_cfg, train_cfg

    try:
        from google.colab import drive

        drive.mount(
            colab_cfg.get("drive_mount_point", "/content/drive"), force_remount=True
        )
    except Exception as e:
        print("Could not mount Google Drive: ", e)
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
