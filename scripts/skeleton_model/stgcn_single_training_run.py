import torch 
from src.config import CHECKPOINT_DIR, DEFAULT_STGCN_PARAMS_V1
from scripts.common.get_device import get_available_device
from scripts.common.seed import set_seed
from src.Skeleton_model.train_model import train_model
from src.Skeleton_model.stgcn import STGCN
from src.Skeleton_model.graph import SkeletonGraph, compute_joint_distance_to_center_of_gravity
from src.rwf2000 import RWF2000PoseDataset
from src.config import POSE_DATASET_ROOT

if __name__ == "__main__":
    run_name="STGCN_run_1" # rename for each run
    save_dir = CHECKPOINT_DIR / "STGCN_V1" / run_name
    params = {} # params to test different from default params
    hyperparameters = DEFAULT_STGCN_PARAMS_V1.copy()
    hyperparameters.update(params)
    set_seed(hyperparameters["seed"])


    device = get_available_device()
    train_dataset = RWF2000PoseDataset(POSE_DATASET_ROOT, split="train", max_people=hyperparameters["max_people"])
    val_dataset = RWF2000PoseDataset(POSE_DATASET_ROOT, split="val", max_people=hyperparameters["max_people"])
    radii = compute_joint_distance_to_center_of_gravity(train_dataset)
    skeleton_graph = SkeletonGraph(radii, normalisation=hyperparameters["adjacency_normalisation_mode"])
    
    model = STGCN(
        skeleton_graph.A,
        temporal_kernel_size=hyperparameters["temporal_kernel_size"], 
        dropout=hyperparameters["dropout"], 
        edge_importance_weighting=hyperparameters["edge_importance_weighting"]
    ).to(device)

    train_model(
        model=model, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        hyperparameters=hyperparameters, 
        device=device, 
        save_dir=save_dir,
        run_name=run_name
    )
