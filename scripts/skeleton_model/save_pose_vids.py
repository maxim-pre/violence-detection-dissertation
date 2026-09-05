import torch
from tqdm import tqdm
from src.config import DATASET_ROOT, POSE_DATASET_ROOT
from src.Skeleton_model.yolo_pose_tracking import save_annotated_pose_videos
from src.rwf2000 import RWF2000PoseDataset, RWF2000Dataset
from ultralytics import YOLO

pose_dataset_root = POSE_DATASET_ROOT
model_name = "yolo26x-pose"

pose_dataset = RWF2000PoseDataset(pose_dataset_root, split="train")
model = YOLO(model_name)

def get_empty_tensors(pose_dataset):

    empty_indices = []
    empty_files = []
    label_map = {
        "Fight": 0,
        "NonFight": 0,
    }

    for index in tqdm(range(len(pose_dataset))):

        skeleton, label = pose_dataset[index]

        if torch.count_nonzero(skeleton) == 0:

            pose_path, _ = pose_dataset.samples[index]

            class_name = "Fight" if label.item() == 1 else "NonFight"

            empty_indices.append(index)
            empty_files.append((pose_path, label))
            label_map[class_name] += 1

    print(f"Total samples: {len(pose_dataset)}")
    print(f"Empty tensors: {len(empty_indices)}")
    print(f"Fight: {label_map['Fight']}")
    print(f"NonFight: {label_map['NonFight']}")

    return empty_files

if __name__ == "__main__":
    empty_files = get_empty_tensors(pose_dataset)

    fight_only_paths = []
    for path, label in empty_files:
        if label == 1: fight_only_paths.append(path)

    save_annotated_pose_videos(model, fight_only_paths, DATASET_ROOT, "train")