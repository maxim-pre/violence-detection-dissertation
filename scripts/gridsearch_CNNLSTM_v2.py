import torch
from src.config import CHECKPOINT_DIR
from scripts.gridsearch_CNNLSTM import grid_search


if __name__ == "__main__":

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V2" / "grid_search_3_weight_decay"

    search_space = [
    {"weight_decay": 0},
    {"weight_decay": 1e-5},
    {"weight_decay": 5e-5},
    {"weight_decay": 1e-4},
    {"weight_decay": 5e-4},
    ]
    
    grid_search(search_space, device, experiment_root=experiment_root, model_version="2")