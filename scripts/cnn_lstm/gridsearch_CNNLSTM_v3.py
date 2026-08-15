import torch
from src.config import CHECKPOINT_DIR
from src.cnn_lstm.gridsearch_CNNLSTM import grid_search
from src.config import DEFAULT_STRONG_AUGMENTATION_PARAMS, DEFAULT_AUGMENTATION_PARAMS
from itertools import product



if __name__ == "__main__":


    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V4" / "test_pooling_modes"

    search_space = [
        {"run_name": "pool_double", "pooling_mode": "double"},
        {"run_name": "pool_avg",    "pooling_mode": "avg"},
        {"run_name": "pool_max",    "pooling_mode": "max"},
        {"run_name": "max_flatten", "pooling_mode": "max_flatten"},
    ]
    
    grid_search(search_space, experiment_root=experiment_root, model_version="4")