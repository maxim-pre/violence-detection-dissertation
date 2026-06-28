from pathlib import Path


# paths
PROJECT_ROOT = Path("/homes/mp2940/violence-detection-dissertation")
DATASET_ROOT = Path("/homes/mp2940/demo/datasets/rwf-2000/RWF-2000")
CHECKPOINT_DIR = Path("/homes/mp2940/violence-detection-dissertation/checkpoints")


DEFAULT_TRAINING_PARAMS_V1 = {
    # optimizer hyperparameters
    "num_frames": 32,
    "augment": True,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "min_lr": 1e-5,
    "factor": 0.5,
    "scheduler_patience": 5,
    "epochs": 75,
    "early_stopping_patience": 15,

    # model hyperparameters
    "hidden_size": 64,
    "num_layers":1,
    "dropout": 0.35,
    "freeze_cnn": True,
    "cnn_cutoff": 16,
    "seed": 42,
}

DEFAULT_TRAINING_PARAMS_V2= {
    # optimizer hyperparameters
    "num_frames": 32,
    "augment": True,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "min_lr": 1e-5,
    "factor": 0.5,
    "scheduler_patience": 5,
    "epochs": 75,
    "early_stopping_patience": 15,
    "seed": 42,

    # model hyperparameters
    "hidden_channels": 64,
    "reduced_channels": 64,
    "dropout": 0.35,
    "freeze_cnn": True,
    "partial_freeze_cnn": False,
    "cnn_cutoff": 16,
}