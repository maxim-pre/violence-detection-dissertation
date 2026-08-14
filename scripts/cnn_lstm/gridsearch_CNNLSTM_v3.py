import torch
from src.config import CHECKPOINT_DIR
from src.cnn_lstm.gridsearch_CNNLSTM import grid_search
from src.config import DEFAULT_STRONG_AUGMENTATION_PARAMS, DEFAULT_AUGMENTATION_PARAMS
from itertools import product



if __name__ == "__main__":


    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V4" / "final_systematic_search_2"

    hidden_channels = [48, 64, 96, 128]
    dropouts = [0.2, 0.25, 0.3, 0.35]
    weight_decays = [0, 5e-6, 1e-5, 2e-5]
    label_smoothing = [0.0, 0.05, 0.1]

    search_space = [
        {
            "run_name": f"h{h}_d{d}_wd{wd}_ls{ls}",
            "hidden_channels": h,
            "dropout": d,
            "weight_decay": wd,
            "label_smoothing": ls,
        }
        for h, d, wd, ls in product(
            hidden_channels,
            dropouts,
            weight_decays,
            label_smoothing,
        )
    ]
    
    grid_search(search_space, experiment_root=experiment_root, model_version="4")