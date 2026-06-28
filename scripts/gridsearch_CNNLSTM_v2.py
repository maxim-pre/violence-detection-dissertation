import torch
from src.config import CHECKPOINT_DIR
from scripts.gridsearch_CNNLSTM import grid_search


if __name__ == "__main__":

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V2" / "grid_search_1"

    search_space = [
        {"hidden_channels": 64,  "reduced_channels": 64,  "dropout": 0.30, "cnn_cutoff": 16},
        {"hidden_channels": 64,  "reduced_channels": 64,  "dropout": 0.35, "cnn_cutoff": 16},
        {"hidden_channels": 128, "reduced_channels": 64,  "dropout": 0.30, "cnn_cutoff": 16},
        {"hidden_channels": 128, "reduced_channels": 64,  "dropout": 0.35, "cnn_cutoff": 16},

        {"hidden_channels": 64,  "reduced_channels": 64,  "dropout": 0.30, "cnn_cutoff": 19},
        {"hidden_channels": 64,  "reduced_channels": 64,  "dropout": 0.35, "cnn_cutoff": 19},
        {"hidden_channels": 128, "reduced_channels": 64,  "dropout": 0.30, "cnn_cutoff": 19},
        {"hidden_channels": 128, "reduced_channels": 64,  "dropout": 0.35, "cnn_cutoff": 19},
        ]
    
    grid_search(search_space, device, experiment_root=experiment_root, model_version="2")