import torch
from src.config import CHECKPOINT_DIR
from src.cnn_lstm.gridsearch_CNNLSTM import grid_search
from src.config import DEFAULT_STRONG_AUGMENTATION_PARAMS, DEFAULT_AUGMENTATION_PARAMS


if __name__ == "__main__":


    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V3" / "grid_search_9_gaussian_blur"

    search_space = [
        {"run_name": "default_aug_unfreeze15", "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "cnn_unfreeze_from": 15},
        {"run_name": "default_aug_unfreeze16", "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "cnn_unfreeze_from": 16},
        {"run_name": "default_aug_unfreeze17", "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "cnn_unfreeze_from": 17},

        {"run_name": "strong_aug_unfreeze15",  "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "cnn_unfreeze_from": 15},
        {"run_name": "strong_aug_unfreeze16",  "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "cnn_unfreeze_from": 16},
        {"run_name": "strong_aug_unfreeze17",  "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "cnn_unfreeze_from": 17},
    ]

    
    grid_search(search_space, experiment_root=experiment_root, model_version="3")