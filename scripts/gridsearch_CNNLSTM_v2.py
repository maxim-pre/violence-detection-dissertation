import torch
from src.config import CHECKPOINT_DIR
from scripts.gridsearch_CNNLSTM import grid_search


if __name__ == "__main__":

    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V3" / "grid_search_2"

    search_space = [
        {
            "run_name": "minlr1e6",
            "min_lr": 1e-6,
        },

        {
            "run_name": "reduced80",
            "reduced_channels": 80,
            "min_lr": 1e-6,
        },
        {
            "run_name": "reduced96",
            "reduced_channels": 96,
            "min_lr": 1e-6,
        },
        {
            "run_name": "reduced112",
            "reduced_channels": 112,
            "min_lr": 1e-6,
        },

        {
            "run_name": "reduced96_lr5e5",
            "reduced_channels": 96,
            "learning_rate": 5e-5,
            "min_lr": 1e-6,
        },
        {
            "run_name": "reduced96_lr75e5",
            "reduced_channels": 96,
            "learning_rate": 7.5e-5,
            "min_lr": 1e-6,
        },

        {
            "run_name": "reduced96_dropout030",
            "reduced_channels": 96,
            "dropout": 0.30,
            "min_lr": 1e-6,
        },
        {
            "run_name": "reduced96_dropout040",
            "reduced_channels": 96,
            "dropout": 0.40,
            "min_lr": 1e-6,
        },
        {
            "run_name": "reduced96_dropout045",
            "reduced_channels": 96,
            "dropout": 0.45,
            "min_lr": 1e-6,
        },

        {
            "run_name": "reduced96_classifier64",
            "reduced_channels": 96,
            "classifier_hidden_size": 64,
            "min_lr": 1e-6,
        },
        {
            "run_name": "reduced96_classifier256",
            "reduced_channels": 96,
            "classifier_hidden_size": 256,
            "min_lr": 1e-6,
        },

        {
            "run_name": "reduced96_patience20",
            "reduced_channels": 96,
            "early_stopping_patience": 20,
            "min_lr": 1e-6,
        },

        {
            "run_name": "reduced96_strong_crop",
            "reduced_channels": 96,
            "min_lr": 1e-6,
            "augmentation_params": {
                "crop_scale_range": (0.75, 1.0),
            },
        },
        {
            "run_name": "reduced96_strong_colour",
            "reduced_channels": 96,
            "min_lr": 1e-6,
            "augmentation_params": {
                "brightness_range": (0.80, 1.20),
                "contrast_range": (0.80, 1.20),
            },
        },
        {
            "run_name": "reduced96_strong_aug",
            "reduced_channels": 96,
            "min_lr": 1e-6,
            "augmentation_params": {
                "crop_scale_range": (0.75, 1.0),
                "brightness_range": (0.80, 1.20),
                "contrast_range": (0.80, 1.20),
            },
        },
    ]
    
    grid_search(search_space, experiment_root=experiment_root, model_version="2")