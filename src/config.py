from pathlib import Path


# paths
PROJECT_ROOT = Path("/homes/mp2940/violence-detection-dissertation")
DATASET_ROOT = Path("/homes/mp2940/demo/datasets/rwf-2000/RWF-2000")
CHECKPOINT_DIR = Path("/homes/mp2940/violence-detection-dissertation/checkpoints")

DEFAULT_AUGMENTATION_PARAMS = {
    "flip_prob": 0.5,
    "brightness_range": (0.85, 1.15),
    "contrast_range": (0.85, 1.15),
    "crop_scale_range": (0.85, 1.0),
}

DEFAULT_STRONG_AUGMENTATION_PARAMS = {
    "flip_prob": 0.5,
    "brightness_range": (0.80, 1.20),
    "contrast_range": (0.80, 1.20),
    "crop_scale_range": (0.75, 1.0),
}

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
    "cnn_cutoff": 19,
    "seed": 42,
}

DEFAULT_TRAINING_PARAMS_V2= {
    # general hyperparameters
    "input_mode": "rgb",
    "num_frames": 32,
    "augment": True,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "cnn_learning_rate": 1e-5,
    "min_lr": 1e-5,
    "factor": 0.5,
    "use_differential_lr": False,
    "scheduler_patience": 5,
    "epochs": 75,
    "early_stopping_patience": 15,
    "weight_decay": 0,
    "amsgrad": False,
    "seed": 42,

    # model hyperparameters
    "hidden_channels": 64,
    "reduced_channels": 64,
    "classifier_hidden_size": 128,
    "dropout": 0.35,
    "cnn_cutoff": 19,
    "cnn_unfreeze_from": None,
}

DEFAULT_TRAINING_PARAMS_V3= {

    #input
    "num_frames": 32,
    "augment": True,
    "batch_size": 4,
    "input_mode": "diff",

    #optimisation
    "learning_rate": 1e-4,
    "cnn_learning_rate": 1e-5,
    "min_lr": 1e-5,
    "cnn_min_lr": 1e-7,
    "factor": 0.5,
    "use_differential_lr": False,
    "scheduler_patience": 5,
    "epochs": 75,
    "early_stopping_patience": 15,
    "weight_decay": 0.0,
    "amsgrad": False,
    "seed": 42,

    # model hyperparameters
    "hidden_channels": 64,
    "reduced_channels": 96,
    "classifier_hidden_size": 128,
    "dropout": 0.35,
    "cnn_cutoff": 19,
    "cnn_unfreeze_from": None,
}