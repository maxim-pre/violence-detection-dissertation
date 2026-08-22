import csv
import time
import torch
from pathlib import Path
from tqdm import tqdm

from src.rwf2000 import RWF2000Dataset
from src.cnn_lstm.cnn_lstm_v3 import CNNLSTMV3
from src.XAI.gradcam import GradCAM
from src.XAI.full_gradcam import FullGradCAM
from src.XAI.smooth_gradcam import SmoothGradCAM
from src.XAI.rise import RISE
from scripts.common.get_device import get_available_device

CHECKPOINT_PATH = Path("PLACEHOLDER/best_model.pt")
OUTPUT_ROOT = Path("/homes/mp2940/demo/datasets/xai_results/cnn_lstm")
RISE_NUM_MASKS = 8000


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
        pooling_mode=config["pooling_mode"],
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


CLASS_NAMES = {0: "NonFight", 1: "Fight"}


def generate_explanations(model, val_dataset, xai_techniques, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    for name in xai_techniques:
        (output_root / name).mkdir(parents=True, exist_ok=True)

    with open(output_root / "predictions.csv", "w", newline="") as predictions_file, \
         open(output_root / "timings.csv", "w", newline="") as timings_file:

        predictions_writer = csv.writer(predictions_file)
        predictions_writer.writerow(["clip_id", "video_path", "true_label", "true_class", "predicted_class", "predicted_class_name", "confidence"])

        timings_writer = csv.writer(timings_file)
        timings_writer.writerow(["clip_id", "technique", "elapsed_seconds"])

        for clip_index in tqdm(range(len(val_dataset)), desc="CNN-LSTM clips"):
            video, label = val_dataset[clip_index]
            input_tensor = video.unsqueeze(0).to(next(model.parameters()).device)

            # use the original video filename (without extension) as the clip id,
            # so saliency maps can always be traced back to the source clip
            video_path, _ = val_dataset.samples[clip_index]
            clip_id = Path(video_path).stem

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
                saliency_map, elapsed_seconds = run_technique(technique, input_tensor, predicted_class)

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
                    output_root / name / f"{clip_id}.pt",
                )

            predictions_file.flush()
            timings_file.flush()


if __name__ == "__main__":
    device = get_available_device()
    model = load_model(CHECKPOINT_PATH, device)

    val_dataset = RWF2000Dataset(
        root_dir=None,  # PLACEHOLDER - set to DATASET_ROOT from src.config
        split="val",
        input_mode="diff",
    )

    xai_techniques = {
        "gradcam": GradCAM(model, target_layer=model.cnn[-1]),
        "multilayer_gradcam": FullGradCAM(model, target_layers=[model.cnn[i] for i in range(19)]),
        "smoothgrad_cam": SmoothGradCAM(model, target_layer=model.cnn[-1]),
        "rise": RISE(model, num_masks=RISE_NUM_MASKS),
    }

    generate_explanations(model, val_dataset, xai_techniques, OUTPUT_ROOT)