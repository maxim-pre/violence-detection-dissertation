import torch
from src.config import CHECKPOINT_DIR
from scripts.gridsearch_CNNLSTM import grid_search


if __name__ == "__main__":

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V2" / "grid_search_5_augmentation_configs"

    search_space = [
        {
            "run_name": "1_default_config",
            "augmentation_params": {
                    "flip_prob": 0.5,
                    "brightness_range": (0.85, 1.15),
                    "contrast_range": (0.85, 1.15),
                    "crop_scale_range": (0.85, 1.0),
            }
        },
        {
            "run_name": "2_stronger_spatial_augmentation",
            "augmentation_params": {
                    "crop_scale_range": (0.75, 1.0),
            }
        },
        {
            "run_name": "3_stronger_brightness_contrast",
            "augmentation_params": {
                    "brightness_range": (0.80, 1.20),
                    "contrast_range": (0.80, 1.20),
            }
        },
        {
            "run_name": "4_stronger_crop_brightness_contrast",
            "augmentation_params": {
                    "crop_scale_range": (0.75, 1.0),
                    "brightness_range": (0.80, 1.20),
                    "contrast_range": (0.80, 1.20),
            }
        },
        {
            "run_name": "5_less_augmentation",
            "augmentation_params": {
                    "crop_scale_range": (0.90, 1.0),
                    "brightness_range": (0.90, 1.10),
                    "contrast_range": (0.90, 1.10),
            }
        },
    ]
    
    grid_search(search_space, device, experiment_root=experiment_root, model_version="2")