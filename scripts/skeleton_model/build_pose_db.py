from src.Skeleton_model.yolo_pose_tracking import build_pose_dataset
from src.rwf2000 import RWF2000Dataset
from src.config import DATASET_ROOT, POSE_DATASET_ROOT
from scripts.common.get_device import get_available_device
from ultralytics import YOLO
import json



if __name__ == "__main__":

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
    POSE_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    print("building training pose dataset...")

    failed_train = build_pose_dataset(
        dataset=val_dataset,
        dataset_root=DATASET_ROOT,
        output_root=POSE_DATASET_ROOT,
        model=model,
        device=device,
    )

    with open("failed_train_pose_extractions.json", "w") as file:
        json.dump(failed_train, file, indent=2)

