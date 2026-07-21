import torch
from src.config import CHECKPOINT_DIR
from scripts.gridsearch_CNNLSTM import grid_search


if __name__ == "__main__":

    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V3" / "grid_search_5_unfreeze_from_15"

    search_space = [
        {"run_name": "TL15_weight_decay_1e4", "use_differential_lr": True, "cnn_unfreeze_from":15, "weight_decay": 1e-4},
        {"run_name": "TL15_weight_decay_5e5", "use_differential_lr": True, "cnn_unfreeze_from":15, "weight_decay": 5e-5},
        {"run_name": "TL15_weight_decay_1e5", "use_differential_lr": True, "cnn_unfreeze_from":15, "weight_decay": 1e-5},
        {"run_name": "TL15_weight_decay_5e6", "use_differential_lr": True, "cnn_unfreeze_from":15, "weight_decay": 5e-6},
        {"run_name": "TL15_cnn_lr_5e6", "use_differential_lr": True, "cnn_unfreeze_from":15, "cnn_learning_rate": 5e-6},
        {"run_name": "TL15_cnn_lr_1e6", "use_differential_lr": True, "cnn_unfreeze_from":15, "cnn_learning_rate": 1e-6},
    ]
    
    grid_search(search_space, experiment_root=experiment_root, model_version="3")