from src.Skeleton_model.yolo_pose_tracking import build_pose_dataset
from src.rwf2000 import RWF2000Dataset
from src.config import DATASET_ROOT, POSE_DATASET_ROOT, GAMMA_DATASET_ROOT, GAMMA_POSE_DATASET_ROOT
from scripts.common.get_device import get_available_device
from ultralytics import YOLO
import json
from pathlib import Path



if __name__ == "__main__":

    new_dataset_root = Path("/homes/mp2940/demo/datasets/rwf-2000/pose_data_ocsort")

    print("building training pose dataset...")

    train_dataset = RWF2000Dataset(
        DATASET_ROOT, 
        split="train", 
        return_rgb_frames=False
    )

    val_dataset = RWF2000Dataset(
        DATASET_ROOT, 
        split="val", 
        return_rgb_frames=False
    )

    model = YOLO("yolo26x-pose")
    device = str(get_available_device())
    new_dataset_root.mkdir(parents=True, exist_ok=True)

    print("building validation pose dataset...")

    failed_train = build_pose_dataset(
        dataset=train_dataset,
        dataset_root=DATASET_ROOT,
        output_root=new_dataset_root,
        model=model,
        device=device,
        tracker="ocsort.yaml"
    )

    with open("failed_train_pose_extractions.json", "w") as file:
        json.dump(failed_train, file, indent=2)

    print("Building validation pose dataset...")

    failed_val = build_pose_dataset(
        dataset=val_dataset,
        dataset_root=DATASET_ROOT,
        output_root=new_dataset_root,
        model=model,
        device=device,
        tracker="ocsort.yaml"
    )

    with open("failed_pose_extractions.json", "w") as file:
        json.dump(failed_val, file, indent=2)

