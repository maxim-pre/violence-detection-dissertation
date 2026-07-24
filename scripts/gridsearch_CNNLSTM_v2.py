import torch
from src.config import CHECKPOINT_DIR
from scripts.gridsearch_CNNLSTM import grid_search
from src.config import DEFAULT_STRONG_AUGMENTATION_PARAMS


if __name__ == "__main__":

    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V3" / "grid_search_6_Freeze_batchnorm"

    search_space = [
        {"run_name": "TL15_Batchnorm_freeze", "use_differential_lr": True, "cnn_unfreeze_from":15},
        {"run_name": "TL15_Batchnorm_freeze_strong_aug", "use_differential_lr": True, "cnn_unfreeze_from":15, "augmentation_params":DEFAULT_STRONG_AUGMENTATION_PARAMS},
        {"run_name": "TL15_Batchnorm_freeze_dropout_40", "use_differential_lr": True, "cnn_unfreeze_from":15, "dropout": 0.4},
        {"run_name": "TL15_Batchnorm_freeze_dropout_45", "use_differential_lr": True, "cnn_unfreeze_from":15, "dropout": 0.45},
    ]
    
    grid_search(search_space, experiment_root=experiment_root, model_version="3")