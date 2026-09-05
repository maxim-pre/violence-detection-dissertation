from pathlib import Path
import shutil
import torch
from tqdm import tqdm
from src.config import POSE_DATASET_ROOT, GAMMA_POSE_DATASET_ROOT, COMBINED_POSE_DATASET_ROOT
from src.rwf2000 import pose_data_to_stgcn_tensor
import json

#---------
# NOT USED
#---------

def score_pose_tensor(tensor):
    # tensor: [3, 150, 17, 4]
    confidence = tensor[2]
    visible = confidence > 0

    person_frames = visible.any(dim=1).sum().item() # number of tracked person appearances across all frames
    visible_joints = visible.sum().item()

    return person_frames, visible_joints

if __name__ == "__main__":

    original = 0 
    gamma = 0 
    ties = 0 

    selected_files = {}

    for path in tqdm(POSE_DATASET_ROOT.rglob("*.pt"), desc="Building combined pose dataset"):

        relative_path = path.relative_to(POSE_DATASET_ROOT)
        gamma_path = GAMMA_POSE_DATASET_ROOT / relative_path
        output_path = COMBINED_POSE_DATASET_ROOT / relative_path

        original_pose = torch.load(path, weights_only=False)
        gamma_pose = torch.load(gamma_path, weights_only=False)

        original_tensor = pose_data_to_stgcn_tensor(original_pose, max_people=4)
        gamma_tensor = pose_data_to_stgcn_tensor(gamma_pose, max_people=4)

        original_score = score_pose_tensor(original_tensor)
        gamma_score = score_pose_tensor(gamma_tensor)

        if gamma_score > original_score:
            selected_path = gamma_path
            gamma+=1
            selected_files[str(relative_path)] = "gamma"

        elif original_score > gamma_score:
            selected_path = path 
            original+=1
            selected_files[str(relative_path)] = "original"
        else:
            selected_path = path 
            ties+=1
            selected_files[str(relative_path)] = "original"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_path, output_path)

    with open(COMBINED_POSE_DATASET_ROOT / "combined_pose_data_files.json", "w") as f:
        json.dump(selected_files, f, indent=2)

    print(f"Original: {original}")
    print(f"Gamma: {gamma}")
    print(f"Ties: {ties}")

        









