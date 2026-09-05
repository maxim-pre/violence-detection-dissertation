import csv
import gc
import time
import torch
from pathlib import Path
from tqdm import tqdm

from src.rwf2000 import RWF2000PoseDataset
from src.Skeleton_model.stgcn import STGCN
from src.config import CHECKPOINT_DIR, POSE_DATASET_ROOT_OCSORT2
from src.XAI.STGCN_grad_cam import STGCNGradCam
from src.XAI.rise import SkeletonRise
from src.XAI.joint_occlusion_2 import JointOcclusion
from scripts.common.get_device import get_available_device

CHECKPOINT_PATH = CHECKPOINT_DIR / "STGCN_V1" / "final_seed_search_updated_masked_bn" / "final_seed42" / "best_model.pt"
OUTPUT_ROOT = Path("/homes/mp2940/demo/datasets/xai_results/stgcn")
RISE_NUM_MASKS = 16000
MAX_CLIPS = None

CLASS_NAMES = {0: "NonFight", 1: "Fight"}


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    placeholder_adjacency = torch.zeros(3, 17, 17)

    model = STGCN(
        placeholder_adjacency,
        in_channels=config.get("in_channels", 3),
        temporal_kernel_size=config["temporal_kernel_size"],
        dropout=config["dropout"],
        edge_importance_weighting=config["edge_importance_weighting"],
        people_aggregation=config.get("people_aggregation", "masked_mean"),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


def run_technique(technique, input_tensor, target_class):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    saliency_map = technique.generate_heatmap(input_tensor, target_class=target_class)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    return saliency_map, elapsed_seconds


def generate_explanations(model, val_dataset, xai_techniques, output_root, max_clips=None):
    output_root.mkdir(parents=True, exist_ok=True)
    for name in xai_techniques:
        (output_root / name).mkdir(parents=True, exist_ok=True)

    predictions_path = output_root / "predictions.csv"
    timings_path = output_root / "timings.csv"
    failures_path = output_root / "failures.csv"

    write_predictions_header = not predictions_path.exists()
    write_timings_header = not timings_path.exists()
    write_failures_header = not failures_path.exists()

    num_clips = len(val_dataset) if max_clips is None else min(max_clips, len(val_dataset))

    with open(predictions_path, "a", newline="") as predictions_file, \
         open(timings_path, "a", newline="") as timings_file, \
         open(failures_path, "a", newline="") as failures_file:

        predictions_writer = csv.writer(predictions_file)
        if write_predictions_header: predictions_writer.writerow(["clip_id", "pose_path", "true_label", "true_class", "predicted_class", "predicted_class_name", "confidence"])

        timings_writer = csv.writer(timings_file)
        if write_timings_header: timings_writer.writerow(["clip_id", "technique", "elapsed_seconds"])

        failures_writer = csv.writer(failures_file)
        if write_failures_header: failures_writer.writerow(["clip_id", "technique", "error"])

        for clip_index in tqdm(range(num_clips), desc="ST-GCN clips"):
            pose_path, _ = val_dataset.samples[clip_index]
            pose_path = Path(pose_path)

            old_clip_id = pose_path.stem  # old naming scheme - kept this to salvage existing results
            clip_id = f"{pose_path.parent.name}_{old_clip_id}"

            # fast path: fully done under the new naming scheme already
            already_done = all((output_root / name / f"{clip_id}.pt").exists() for name in xai_techniques)
            if already_done: continue

            skeleton, label = val_dataset[clip_index]
            input_tensor = skeleton.unsqueeze(0).to(next(model.parameters()).device)

            true_label = label.item()
            true_class_name = CLASS_NAMES[true_label]

            with torch.inference_mode():
                probabilities = torch.softmax(model(input_tensor), dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            predicted_class_name = CLASS_NAMES[predicted_class]
            confidence = probabilities[0, predicted_class].item()

            predictions_writer.writerow([
                clip_id, str(pose_path), true_label, true_class_name,
                predicted_class, predicted_class_name, confidence,
            ])

            for name, technique in xai_techniques.items():
                new_output_path = output_root / name / f"{clip_id}.pt"
                old_output_path = output_root / name / f"{old_clip_id}.pt"

                if new_output_path.exists(): continue 

                if old_output_path.exists():
                    saved = torch.load(old_output_path, map_location="cpu", weights_only=False)
                    if Path(saved["pose_path"]) == pose_path:
                        old_output_path.rename(new_output_path)
                        continue

                try:
                    saliency_map, elapsed_seconds = run_technique(technique, input_tensor, predicted_class)
                except torch.cuda.OutOfMemoryError as error:
                    failures_writer.writerow([clip_id, name, str(error)])
                    failures_file.flush()
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue  # skip this technique for this clip instead of crashing the whole run

                timings_writer.writerow([clip_id, name, elapsed_seconds])
                torch.save(
                    {
                        "saliency_map": saliency_map.cpu(),
                        "pose_path": str(pose_path),
                        "true_label": true_label,
                        "true_class_name": true_class_name,
                        "predicted_class": predicted_class,
                        "predicted_class_name": predicted_class_name,
                        "elapsed_seconds": elapsed_seconds,
                    },
                    new_output_path,
                )

            predictions_file.flush()
            timings_file.flush()


if __name__ == "__main__":
    device = get_available_device()
    model, config = load_model(CHECKPOINT_PATH, device)

    val_dataset = RWF2000PoseDataset(
        root_dir=POSE_DATASET_ROOT_OCSORT2,
        split="val",
        max_people=config.get("max_people", 4),
        score_mode=config.get("score_mode", "total_confidence"),
    )

    xai_techniques = {
        "stg_gradcam": STGCNGradCam(model, target_layer=model.stgcn_blocks[-1], normalisation_mode="per_video"),
        "rise": SkeletonRise(model, num_masks=RISE_NUM_MASKS, normalisation_mode="per_video"),
        "joint_occlusion": JointOcclusion(model, normalisation_mode="per_video"),
    }

    generate_explanations(
        model=model,
        val_dataset=val_dataset,
        xai_techniques=xai_techniques,
        output_root=OUTPUT_ROOT,
        max_clips=MAX_CLIPS,
    )