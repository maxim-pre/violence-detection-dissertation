import torch
from src.config import CHECKPOINT_DIR
from scripts.gridsearch_CNNLSTM import grid_search


if __name__ == "__main__":

    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V3" / "grid_search_1"

    search_space = [
        # baseline reference
        {
            "run_name": "baseline"
        },

        # temporal context
        {
            "run_name": "frames_48",
            "num_frames": 48
        },
        {
            "run_name": "frames_64",
            "num_frames": 64
        },

        # dropout
        {
            "run_name": "dropout_030",
            "dropout": 0.30
        },
        {
            "run_name": "dropout_040",
            "dropout": 0.40
        },
        {
            "run_name": "dropout_045",
            "dropout": 0.45
        },

        # SepConvLSTM capacity
        {
            "run_name": "hidden_96",
            "hidden_channels": 96
        },
        {
            "run_name": "hidden_128",
            "hidden_channels": 128
        },

        # CNN channel reduction
        {
            "run_name": "reduced_96",
            "reduced_channels": 96
        },
        {
            "run_name": "reduced_128",
            "reduced_channels": 128
        },

        # classifier capacity
        {
            "run_name": "classifier_64",
            "classifier_hidden_size": 64
        },
        {
            "run_name": "classifier_256",
            "classifier_hidden_size": 256
        },

        # optimiser
        {
            "run_name": "lr_5e-5",
            "learning_rate": 5e-5
        },
        {
            "run_name": "lr_2e-4",
            "learning_rate": 2e-4
        },

        # combined promising variants
        {
            "run_name": "frames48_hidden96",
            "num_frames": 48,
            "hidden_channels": 96
        },
        {
            "run_name": "frames48_reduced96",
            "num_frames": 48,
            "reduced_channels": 96
        },
        {
            "run_name": "frames64_dropout40",
            "num_frames": 64,
            "dropout": 0.40
        },
        {
            "run_name": "hidden96_dropout40",
            "hidden_channels": 96,
            "dropout": 0.40
        },
        {
            "run_name": "reduced96_dropout40",
            "reduced_channels": 96,
            "dropout": 0.40
        },

        # augmentation
        {
            "run_name": "strong_crop",
            "augmentation_params": {
                "crop_scale_range": (0.75, 1.0)
            }
        },
        {
            "run_name": "strong_colour",
            "augmentation_params": {
                "brightness_range": (0.80, 1.20),
                "contrast_range": (0.80, 1.20)
            }
        },
        {
            "run_name": "strong_aug",
            "augmentation_params": {
                "crop_scale_range": (0.75, 1.0),
                "brightness_range": (0.80, 1.20),
                "contrast_range": (0.80, 1.20)
            }
        },
    ]
    
    grid_search(search_space, experiment_root=experiment_root, model_version="2")