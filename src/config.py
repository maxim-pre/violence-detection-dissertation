from pathlib import Path


# paths
PROJECT_ROOT = Path("/homes/mp2940/violence-detection-dissertation")
DATASET_ROOT = Path("/homes/mp2940/demo/datasets/rwf-2000/RWF-2000")
POSE_DATASET_ROOT = Path("/homes/mp2940/demo/datasets/rwf-2000/pose_data")
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
    "seed": 42,

    #input
    "num_frames": 32,
    "augment": True,
    "batch_size": 4,
    "input_mode": "diff",

    #optimisation
    "learning_rate": 1e-4,
    "cnn_learning_rate": 1e-6,
    "min_lr": 1e-5,
    "cnn_min_lr": 1e-7,
    "factor": 0.5,
    "use_differential_lr": True,
    "scheduler_patience": 5,
    "epochs": 75,
    "early_stopping_patience": 15,
    "weight_decay": 0.0,
    "amsgrad": False,
    "label_smoothing": 0.0,

    # model hyperparameters
    "hidden_channels": 64,
    "reduced_channels": 96,
    "classifier_hidden_size": 128,
    "dropout": 0.35,
    "cnn_cutoff": 19,
    "cnn_unfreeze_from": 15,
    "freeze_cnn_batchnorm": False,
}

DEFAULT_STGCN_PARAMS_V1 = {
    "seed": 42, 

    # input
    "max_people": 4, 
    "batch_size": 4, 

    # augmentation
    "augment": False,

    # optimisation
    "epochs": 80,
    "learning_rate": 2e-4, 
    "min_lr": 1e-6, 
    "scheduler_patience": 10, 
    "factor": 0.5, 
    "early_stopping_patience": 15,
    "amsgrad": False, 
    "weight_decay": 1e-4, 
    "label_smoothing": 0.0,

    # model
    "adjacency_normalisation_mode": "column", 
    "temporal_kernel_size": 9, 
    "dropout": 0.3, 
    "edge_importance_weighting": True
}

DEFAULT_POSE_AUGMENTATION_PARAMS = {
    "whole_occlusion_min_frames": 10,
    "whole_occlusion_max_frames": 25,
    "whole_occlusion_probability": 0.5,

    "mirror_min_frames": 1,
    "mirror_max_frames": 4,
    "mirror_probability": 0.5,
}