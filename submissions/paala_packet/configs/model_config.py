# configs/model_config.py
import torch

CONFIG_A = {
    "data_dir":        "data/hls_burn_scars",
    "model_save_dir":  "models",
    "output_dir":      "outputs",
    "batch_size":      1,
    "num_epochs":      20,
    "learning_rate":   1e-4,
    "image_size":      224,
    "num_workers":     0,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "freeze_encoder":  True,
}

CONFIG_B = {
    "data_dir":       "data/ndws",
    "model_save_dir": "models",
    "output_dir":     "outputs",
    "batch_size":     32,
    "num_epochs":     50,
    "learning_rate":  3e-4,
    "image_size":     64,
    "num_workers":    0,
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "iou_threshold":  0.3,
}