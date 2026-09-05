import torch
from src.config import CHECKPOINT_DIR
from src.cnn_lstm.gridsearch_CNNLSTM import grid_search
from src.config import DEFAULT_STRONG_AUGMENTATION_PARAMS, DEFAULT_AUGMENTATION_PARAMS
from itertools import product



if __name__ == "__main__":


    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V4" / "Final_models"

    search_space = [
        {"run_name": "rgb", "input_mode": "rgb"},
        {"run_name": "frame_diff", "input_mode": "diff"},
    ]
    
    grid_search(search_space, experiment_root=experiment_root, model_version="4")