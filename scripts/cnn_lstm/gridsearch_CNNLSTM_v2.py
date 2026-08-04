import torch
from src.config import CHECKPOINT_DIR
from src.cnn_lstm.gridsearch_CNNLSTM import grid_search
from src.config import DEFAULT_STRONG_AUGMENTATION_PARAMS, DEFAULT_AUGMENTATION_PARAMS


if __name__ == "__main__":

    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V3" / "grid_search_7"

    search_space = [
    {"run_name": "default_wd1e6_ls0_bntrain",    "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "weight_decay": 1e-6, "label_smoothing": 0.0, "freeze_cnn_batchnorm": False},
    {"run_name": "default_wd1e6_ls0_bnfrozen",   "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "weight_decay": 1e-6, "label_smoothing": 0.0, "freeze_cnn_batchnorm": True},
    {"run_name": "default_wd1e6_ls01_bntrain",   "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "weight_decay": 1e-6, "label_smoothing": 0.1, "freeze_cnn_batchnorm": False},
    {"run_name": "default_wd1e6_ls01_bnfrozen",  "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "weight_decay": 1e-6, "label_smoothing": 0.1, "freeze_cnn_batchnorm": True},

    {"run_name": "default_wd5e6_ls0_bntrain",    "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "weight_decay": 5e-6, "label_smoothing": 0.0, "freeze_cnn_batchnorm": False},
    {"run_name": "default_wd5e6_ls0_bnfrozen",   "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "weight_decay": 5e-6, "label_smoothing": 0.0, "freeze_cnn_batchnorm": True},
    {"run_name": "default_wd5e6_ls01_bntrain",   "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "weight_decay": 5e-6, "label_smoothing": 0.1, "freeze_cnn_batchnorm": False},
    {"run_name": "default_wd5e6_ls01_bnfrozen",  "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "weight_decay": 5e-6, "label_smoothing": 0.1, "freeze_cnn_batchnorm": True},

    {"run_name": "default_wd1e5_ls0_bntrain",    "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "weight_decay": 1e-5, "label_smoothing": 0.0, "freeze_cnn_batchnorm": False},
    {"run_name": "default_wd1e5_ls0_bnfrozen",   "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "weight_decay": 1e-5, "label_smoothing": 0.0, "freeze_cnn_batchnorm": True},
    {"run_name": "default_wd1e5_ls01_bntrain",   "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "weight_decay": 1e-5, "label_smoothing": 0.1, "freeze_cnn_batchnorm": False},
    {"run_name": "default_wd1e5_ls01_bnfrozen",  "augmentation_params": DEFAULT_AUGMENTATION_PARAMS,        "weight_decay": 1e-5, "label_smoothing": 0.1, "freeze_cnn_batchnorm": True},

    # ---------------- Strong augmentation ----------------
    {"run_name": "strong_wd1e6_ls0_bntrain",     "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "weight_decay": 1e-6, "label_smoothing": 0.0, "freeze_cnn_batchnorm": False},
    {"run_name": "strong_wd1e6_ls0_bnfrozen",    "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "weight_decay": 1e-6, "label_smoothing": 0.0, "freeze_cnn_batchnorm": True},
    {"run_name": "strong_wd1e6_ls01_bntrain",    "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "weight_decay": 1e-6, "label_smoothing": 0.1, "freeze_cnn_batchnorm": False},
    {"run_name": "strong_wd1e6_ls01_bnfrozen",   "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "weight_decay": 1e-6, "label_smoothing": 0.1, "freeze_cnn_batchnorm": True},

    {"run_name": "strong_wd5e6_ls0_bntrain",     "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "weight_decay": 5e-6, "label_smoothing": 0.0, "freeze_cnn_batchnorm": False},
    {"run_name": "strong_wd5e6_ls0_bnfrozen",    "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "weight_decay": 5e-6, "label_smoothing": 0.0, "freeze_cnn_batchnorm": True},
    {"run_name": "strong_wd5e6_ls01_bntrain",    "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "weight_decay": 5e-6, "label_smoothing": 0.1, "freeze_cnn_batchnorm": False},
    {"run_name": "strong_wd5e6_ls01_bnfrozen",   "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "weight_decay": 5e-6, "label_smoothing": 0.1, "freeze_cnn_batchnorm": True},

    {"run_name": "strong_wd1e5_ls0_bntrain",     "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "weight_decay": 1e-5, "label_smoothing": 0.0, "freeze_cnn_batchnorm": False},
    {"run_name": "strong_wd1e5_ls0_bnfrozen",    "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "weight_decay": 1e-5, "label_smoothing": 0.0, "freeze_cnn_batchnorm": True},
    {"run_name": "strong_wd1e5_ls01_bntrain",    "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "weight_decay": 1e-5, "label_smoothing": 0.1, "freeze_cnn_batchnorm": False},
    {"run_name": "strong_wd1e5_ls01_bnfrozen",   "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, "weight_decay": 1e-5, "label_smoothing": 0.1, "freeze_cnn_batchnorm": True},
    ]
    
    grid_search(search_space, experiment_root=experiment_root, model_version="3")