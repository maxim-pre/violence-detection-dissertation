import torch
from src.config import CHECKPOINT_DIR
from scripts.gridsearch_CNNLSTM import grid_search


if __name__ == "__main__":

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V2" / "grid_search_2_unfreeze_cnn"

    search_space = [
    {"cnn_unfreeze_from": None},
    {"cnn_unfreeze_from": 17, "use_differential_lr": True, "cnn_learning_rate": 1e-5},
    {"cnn_unfreeze_from": 15, "use_differential_lr": True, "cnn_learning_rate": 1e-5},
    {"cnn_unfreeze_from": 13, "use_differential_lr": True, "cnn_learning_rate": 1e-5},
    ]
    
    grid_search(search_space, device, experiment_root=experiment_root, model_version="2")