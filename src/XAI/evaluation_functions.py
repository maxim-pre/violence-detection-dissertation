import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TECHNIQUE_DISPLAY_NAMES = {
    "stg_gradcam": "STG-Grad-CAM",
    "rise": "RISE",
    "joint_occlusion": "Joint Occlusion",
}

# mean saliency per frame
def compute_frame_totals(sample, pose_tensor=None):
    saliency_map = sample["saliency_map"].squeeze()  
    frame_totals = saliency_map.flatten(1).sum(dim=1)  # [T]

    if pose_tensor is not None:
        confidence = pose_tensor[0, 2]  # [T, V, M]
        people_per_frame = (confidence.sum(dim=1) > 0).sum(dim=1)  # [T]
        frame_totals = frame_totals / people_per_frame.clamp(min=1)

    return frame_totals


def compute_mean_saliency_per_frame(xai_results_root, technique, predicted_class_name=None, exclude_clip_ids=None,
                                     val_dataset=None, pose_path_to_index=None):
    technique_dir = xai_results_root / technique
    per_clip_frame_totals = []

    for clip_path in sorted(technique_dir.glob("*.pt")):
        if exclude_clip_ids is not None and clip_path.stem in exclude_clip_ids:
            continue

        sample = torch.load(clip_path, weights_only=False)

        if predicted_class_name is not None and sample["predicted_class_name"] != predicted_class_name:
            continue

        pose_tensor = None
        if val_dataset is not None:
            index = pose_path_to_index[sample["pose_path"]]
            skeleton, _ = val_dataset[index]
            pose_tensor = skeleton.unsqueeze(0)

        per_clip_frame_totals.append(compute_frame_totals(sample, pose_tensor=pose_tensor))

    stacked = torch.stack(per_clip_frame_totals)  # [num_clips_matching, T]
    mean_per_frame = stacked.mean(dim=0)  # [T]

    return mean_per_frame


def plot_saliency_per_frame(fight, nonfight, title, ylabel="untitled", save=False, save_path=None):
    fig = plt.figure(figsize=(10, 6))

    for technique, values in fight.items():
        label = TECHNIQUE_DISPLAY_NAMES.get(technique, technique)

        line, = plt.plot(
            values.numpy(),
            linewidth=2,
            linestyle="-",
            label=label
        )

        plt.plot(
            nonfight[technique].numpy(),
            linewidth=2,
            linestyle="--",
            color=line.get_color(),
            alpha=0.8
        )

    plt.xlabel("Frame", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.tight_layout()

    if save:
        if save_path is None:
            raise ValueError("save_path must be provided when save=True")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



# Saliency distribution -------------------------------


def collect_saliency_values(technique_dir, exclude_clip_ids, val_dataset, pose_path_to_index, predicted_class_name=None):
    values = []

    for clip_path in sorted(technique_dir.glob("*.pt")):
        if clip_path.stem in exclude_clip_ids:
            continue

        sample = torch.load(clip_path, weights_only=False)

        if predicted_class_name is not None and sample["predicted_class_name"] != predicted_class_name:
            continue

        saliency = sample["saliency_map"].squeeze()  # [T, V, M]

        index = pose_path_to_index[sample["pose_path"]]
        skeleton, _ = val_dataset[index]
        detected = skeleton[2] > 0  # [T, V, M]
        values.append(saliency[detected])

    return torch.cat(values)

def plot_saliency_distribution(values_by_technique, title, bins=50, save=False, save_dir=None):
    for technique, values in values_by_technique.items():
        display_name = TECHNIQUE_DISPLAY_NAMES.get(technique, technique)
        values = values.numpy()
        weights = np.full_like(values, 100 / len(values))

        plt.figure(figsize=(6, 5))
        plt.hist(values, bins=bins, weights=weights)
        plt.xlabel("Saliency value", fontsize=12)
        plt.ylabel("% of values", fontsize=12)
        plt.title(f"{title} — {display_name}", fontsize=14)
        plt.tight_layout()

        if save:
            if save_dir is None:
                raise ValueError("save_dir must be provided when save=True")
            plt.savefig(save_dir / f"{technique}_saliency_distribution.png", dpi=300, bbox_inches="tight")

        plt.show()


# per clip mean saliency 

def collect_per_clip_mean_saliency(technique_dir, predictions_df, val_dataset, pose_path_to_index):
    records = []

    for _, row in predictions_df.iterrows():
        clip_path = technique_dir / f"{row['clip_id']}.pt"
        if not clip_path.exists():
            continue

        sample = torch.load(clip_path, weights_only=False)
        saliency = sample["saliency_map"].squeeze()

        index = pose_path_to_index[row["pose_path"]]
        skeleton, _ = val_dataset[index]
        detected = skeleton[2] > 0  # [T, V, M]
        mean_saliency = saliency[detected].mean().item()

        person_present = detected.any(dim=1)      # [T, M]
        max_people = person_present.sum(dim=1).max().item()

        records.append({
            "clip_id": row["clip_id"],
            "mean_saliency": mean_saliency,
            "confidence": row["confidence"],
            "max_people": max_people,
        })

    return pd.DataFrame(records)