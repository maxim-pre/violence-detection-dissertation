import torch
from src.config import CHECKPOINT_DIR
from src.cnn_lstm.gridsearch_CNNLSTM import grid_search
from src.config import DEFAULT_STRONG_AUGMENTATION_PARAMS, DEFAULT_AUGMENTATION_PARAMS


if __name__ == "__main__":


    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V3" / "grid_search_10"

    search_space = [
        {"run_name": "baseline",          "hidden_channels": 64, "reduced_channels": 96},
        {"run_name": "hidden48",          "hidden_channels": 48, "reduced_channels": 96},
        {"run_name": "hidden32",          "hidden_channels": 32, "reduced_channels": 96},
        {"run_name": "reduced64",         "hidden_channels": 64, "reduced_channels": 64},
        {"run_name": "hidden48_red64",    "hidden_channels": 48, "reduced_channels": 64},
    ]
    
    grid_search(search_space, experiment_root=experiment_root, model_version="3")