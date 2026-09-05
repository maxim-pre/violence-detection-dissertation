from pathlib import Path


# paths
PROJECT_ROOT = Path("/homes/mp2940/violence-detection-dissertation")

DATASET_ROOT = Path("/homes/mp2940/demo/datasets/rwf-2000/RWF-2000")
GAMMA_DATASET_ROOT = Path("/homes/mp2940/demo/datasets/rwf-2000/RWF-2000-gamma-067")

# persist=False
POSE_DATASET_ROOT_BYTETRACK = Path("/homes/mp2940/demo/datasets/rwf-2000/pose_data_bytetrack")
POSE_DATASET_ROOT_OCSORT2 = Path("/homes/mp2940/demo/datasets/rwf-2000/pose_data_ocsort2")

# persist=True
POSE_DATASET_ROOT = Path("/homes/mp2940/demo/datasets/rwf-2000/pose_data")
POSE_DATASET_ROOT_OCSORT = Path("/homes/mp2940/demo/datasets/rwf-2000/pose_data_ocsort")
GAMMA_POSE_DATASET_ROOT = Path("/homes/mp2940/demo/datasets/rwf-2000/pose_data_gamma_067")
COMBINED_POSE_DATASET_ROOT = Path("/homes/mp2940/demo/datasets/rwf-2000/pose_data_combined")

# XAI results paths
CNN_LSTM_XAI_RESULTS = Path("/homes/mp2940/demo/datasets/xai_results/cnn_lstm")
STGCN_XAI_RESULTS = Path("/homes/mp2940/demo/datasets/xai_results/stgcn")


CHECKPOINT_DIR = Path("/homes/mp2940/violence-detection-dissertation/checkpoints")

DEFAULT_AUGMENTATION_PARAMS = {
    "flip_prob": 0.5,
    "brightness_range": (0.85, 1.15),
    "contrast_range": (0.85, 1.15),
    "crop_scale_range": (0.85, 1.0),
    "gaussian_blur_prob": 0.1,
    "gaussian_blur_sigma_range": (0.1, 1.0),
}

DEFAULT_STRONG_AUGMENTATION_PARAMS = {
    "flip_prob": 0.5,
    "brightness_range": (0.80, 1.20),
    "contrast_range": (0.80, 1.20),
    "crop_scale_range": (0.75, 1.0),
    "gaussian_blur_prob": 0.1,
    "gaussian_blur_sigma_range": (0.1, 1.5),
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
    "use_differential_lr": False,
    "scheduler_patience": 5,
    "epochs": 75,
    "early_stopping_patience": 15,
    "weight_decay": 5e-6,
    "amsgrad": False,
    "label_smoothing": 0.1,

    # model hyperparameters
    "hidden_channels": 64,
    "reduced_channels": 96,
    "classifier_hidden_size": 128,
    "dropout": 0.35,
    "cnn_cutoff": 19,
    "cnn_unfreeze_from": None,
    "freeze_cnn_batchnorm": True,

    "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, 
}

DEFAULT_TRAINING_PARAMS_V4= {
    "seed": 42,

    #input
    "num_frames": 32,
    "augment": True,
    "batch_size": 8,
    "input_mode": "diff",

    #optimisation
    "learning_rate": 5e-5,
    "cnn_learning_rate": 1e-6,
    "min_lr": 1e-7,
    "cnn_min_lr": 1e-7,
    "factor": 0.5,
    "use_differential_lr": False,
    "scheduler_patience": 5,
    "epochs": 75,
    "early_stopping_patience": 8,
    "weight_decay": 2e-5,
    "amsgrad": False,
    "label_smoothing": 0.1,

    # model hyperparameters
    "hidden_channels": 64,
    "reduced_channels": 96,
    "classifier_hidden_size": 128,
    "dropout": 0.35,
    "cnn_cutoff": 19,
    "cnn_unfreeze_from": None,
    "freeze_cnn_batchnorm": True,
    "pooling_mode": "double",

    "augmentation_params": DEFAULT_STRONG_AUGMENTATION_PARAMS, 
}

DEFAULT_STGCN_PARAMS_V1 = {
    "seed": 45, 

    # input
    "use_gamma_corrected_data": False, 
    "max_people": 4, 
    "score_mode": "total_confidence",
    "batch_size": 12, 

    # augmentation
    "augment": True,

    # optimisation
    "epochs": 60,
    "learning_rate": 3e-4, 
    "min_lr": 1e-6, 
    "scheduler_patience": 5, 
    "factor": 0.5, 
    "early_stopping_patience": 20,
    "amsgrad": False, 
    "weight_decay": 5e-5, 
    "label_smoothing": 0.2,

    # model
    "adjacency_normalisation_mode": "column", 
    "temporal_kernel_size": 9, 
    "dropout": 0.3, 
    "edge_importance_weighting": True,
    "people_aggregation": "masked_mean",
}

DEFAULT_POSE_AUGMENTATION_PARAMS = {
    # frame occlusion
    "whole_occlusion_min_frames": 10,
    "whole_occlusion_max_frames": 25,
    "whole_occlusion_probability": 0.5,

    # Body part interpolation
    "body_part_interpolation_min_frames": 1,
    "body_part_interpolation_max_frames": 10,
    "body_part_interpolation_probability": 0.65,

    # Whole-skeleton interpolation
    "interpolation_min_frames": 1,
    "interpolation_max_frames": 10,
    "interpolation_probability": 0.65,

    # Random keypoint swapping
    "swap_min_frames": 1,
    "swap_max_frames": 4,
    "swap_probability": 0.5,

    # mirroring
    "mirror_min_frames": 1,
    "mirror_max_frames": 4,
    "mirror_probability": 0.5,
}

