import torch
from src.config import CHECKPOINT_DIR
from src.cnn_lstm.gridsearch_CNNLSTM import grid_search
from src.config import DEFAULT_STRONG_AUGMENTATION_PARAMS, DEFAULT_AUGMENTATION_PARAMS


if __name__ == "__main__":


    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V4" / "regularisation_search"

    search_space = [
        {"run_name": "baseline"},

        {"run_name": "lr_5e-5", "learning_rate": 5e-5},
        {"run_name": "lr_2e-4", "learning_rate": 2e-4},
        {"run_name": "lr_3e-4", "learning_rate": 3e-4},

        {"run_name": "dropout_0.25", "dropout": 0.25},
        {"run_name": "dropout_0.45", "dropout": 0.45},
        {"run_name": "dropout_0.55", "dropout": 0.55},

        {"run_name": "wd_1e-6", "weight_decay": 1e-6},
        {"run_name": "wd_1e-5", "weight_decay": 1e-5},
        {"run_name": "wd_5e-5", "weight_decay": 5e-5},
        {"run_name": "wd_1e-4", "weight_decay": 1e-4},

        {"run_name": "ls_0.0", "label_smoothing": 0.0},
        {"run_name": "ls_0.05", "label_smoothing": 0.05},
        {"run_name": "ls_0.15", "label_smoothing": 0.15},

        {"run_name": "reg_1", "dropout": 0.45, "weight_decay": 1e-5},
        {"run_name": "reg_2", "dropout": 0.45, "weight_decay": 5e-5},
        {"run_name": "reg_3", "dropout": 0.55, "weight_decay": 5e-5},
        {"run_name": "reg_4", "dropout": 0.45, "weight_decay": 1e-5, "label_smoothing": 0.05},
        {"run_name": "best_guess", "learning_rate": 5e-5, "dropout": 0.45, "weight_decay": 1e-5},
    ]
    
    grid_search(search_space, experiment_root=experiment_root, model_version="4")