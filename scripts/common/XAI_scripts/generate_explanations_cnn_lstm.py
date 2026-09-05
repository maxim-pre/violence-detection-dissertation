import csv
import gc
import time
import torch
import json
from pathlib import Path
from tqdm import tqdm

from src.rwf2000 import RWF2000Dataset
from src.cnn_lstm.cnn_lstm_v3 import CNNLSTMV3
from src.config import CHECKPOINT_DIR, DATASET_ROOT
from src.XAI.gradcam import GradCAM
from src.XAI.full_gradcam import FullGradCAM
from src.XAI.smooth_gradcam import SmoothGradCAM
from src.XAI.rise import RISE
from scripts.common.get_device import get_available_device

CHECKPOINT_PATH = CHECKPOINT_DIR / "CNN_LSTM_V4" / "final_systematic_search_2" / "h64_d0.35_wd2e-05_ls0.1" / "best_model.pt"
OUTPUT_ROOT = Path("/homes/mp2940/demo/datasets/xai_results/cnn_lstm")
RISE_NUM_MASKS = 16000
MAX_CLIPS = None

CLASS_NAMES = {0: "NonFight", 1: "Fight"}


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = CNNLSTMV3(
        hidden_channels=config["hidden_channels"],
        reduced_channels=config["reduced_channels"],
        classifier_hidden_size=config["classifier_hidden_size"],
        dropout=config["dropout"],
        cnn_cutoff=config["cnn_cutoff"],
        cnn_unfreeze_from=config["cnn_unfreeze_from"],
        pooling_mode=config.get("pooling_mode", 'double'),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def run_technique(technique, input_tensor, target_class):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    saliency_map = technique.generate_heatmap(input_tensor, target_class=target_class)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    return saliency_map, elapsed_seconds


def generate_explanations(model, val_dataset, xai_techniques, output_root, max_clips=None, target_clip_ids=None):
    output_root.mkdir(parents=True, exist_ok=True)
    for name in xai_techniques: (output_root / name).mkdir(parents=True, exist_ok=True)

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
        if write_predictions_header: predictions_writer.writerow(["clip_id", "video_path", "true_label", "true_class", "predicted_class", "predicted_class_name", "confidence"])

        timings_writer = csv.writer(timings_file)
        if write_timings_header: timings_writer.writerow(["clip_id", "technique", "elapsed_seconds"])

        failures_writer = csv.writer(failures_file)
        if write_failures_header: failures_writer.writerow(["clip_id", "technique", "error"])

        for clip_index in tqdm(range(num_clips), desc="CNN-LSTM clips"):
            video_path, _ = val_dataset.samples[clip_index]
            video_path = Path(video_path)
 
            old_clip_id = video_path.stem  # old naming scheme - kept this to salvage existing results
            clip_id = f"{video_path.parent.name}_{old_clip_id}"

            if target_clip_ids is not None and clip_id not in target_clip_ids: continue

            # skip this clip entirely if every technique already has a saved output for it
            already_done = all((output_root / name / f"{clip_id}.pt").exists() for name in xai_techniques)
            if already_done: continue

            video, label = val_dataset[clip_index]
            input_tensor = video.unsqueeze(0).to(next(model.parameters()).device)

            true_label = label.item()
            true_class_name = CLASS_NAMES[true_label]

            with torch.inference_mode():
                probabilities = torch.softmax(model(input_tensor), dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            predicted_class_name = CLASS_NAMES[predicted_class]
            confidence = probabilities[0, predicted_class].item()

            predictions_writer.writerow([
                clip_id, str(video_path), true_label, true_class_name,
                predicted_class, predicted_class_name, confidence,
            ])

            for name, technique in xai_techniques.items():
                new_output_path = output_root / name / f"{clip_id}.pt"
                old_output_path = output_root / name / f"{old_clip_id}.pt"

                if new_output_path.exists():
                    continue  

                if old_output_path.exists():
                    saved = torch.load(old_output_path, map_location="cpu", weights_only=False)
                    if Path(saved["video_path"]) == video_path:
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
                        "video_path": str(video_path),
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
    model = load_model(CHECKPOINT_PATH, device)

    val_dataset = RWF2000Dataset(
        root_dir=DATASET_ROOT,
        split="val",
        input_mode="diff",
    )

    xai_techniques = {
        "gradcam": GradCAM(model, target_layer=model.cnn[-1]),
        "multilayer_gradcam": FullGradCAM(model, target_layers=[model.cnn[i] for i in range(19)]),
        "smoothgrad_cam": SmoothGradCAM(model, target_layer=model.cnn[-1]),
        "rise": RISE(model, num_masks=RISE_NUM_MASKS),
    }

    generate_explanations(
        model=model,
        val_dataset=val_dataset,
        xai_techniques=xai_techniques,
        output_root=OUTPUT_ROOT,
        max_clips=MAX_CLIPS,
    )