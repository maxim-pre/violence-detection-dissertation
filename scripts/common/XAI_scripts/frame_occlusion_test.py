import torch
from tqdm import tqdm
from scripts.common.get_device import get_available_device
from src.config import DATASET_ROOT, CNN_LSTM_XAI_RESULTS, CHECKPOINT_DIR
from src.rwf2000 import RWF2000Dataset
from src.cnn_lstm.cnn_lstm_v3 import CNNLSTMV3

NUM_FRAMES = 32
BASELINE_VALUE = 0.0
DEVICE = get_available_device()

RESULTS_DIR = CNN_LSTM_XAI_RESULTS / "frame_oclusuion_results"

def load_final_cnn_lstm_model():
    checkpoint_path = CHECKPOINT_DIR / "CNN_LSTM_V4" / "final_systematic_search_2" / "h64_d0.35_wd2e-05_ls0.1" / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    model = CNNLSTMV3() 
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model


def measure_frame_importance_by_occlusion(model, video_tensor, target_class, baseline_value=BASELINE_VALUE):

    with torch.inference_mode():
        original_logits = model(video_tensor)
        original_confidence = torch.softmax(original_logits, dim=1)[0, target_class].item()

    T = video_tensor.shape[1]
    confidence_drops = torch.zeros(T)

    for t in range(T):
        occluded = video_tensor.clone()
        occluded[:, t] = baseline_value

        with torch.inference_mode():
            occluded_logits = model(occluded)
            occluded_confidence = torch.softmax(occluded_logits, dim=1)[0, target_class].item()

        confidence_drops[t] = original_confidence - occluded_confidence

    return confidence_drops


def run_test(model, gradcam_results_dir, dataset_root, output_dir, predicted_class_name=None):

    val_dataset = RWF2000Dataset(
        dataset_root,
        split="val",
        augment=False,
        input_mode="diff",
    )

    video_path_to_index = {str(video_path): idx for idx, (video_path, label) in enumerate(val_dataset.samples)}
    output_dir.mkdir(parents=True, exist_ok=True)
    per_clip_drops = []

    clip_paths = sorted(gradcam_results_dir.glob("*.pt"))

    for clip_path in tqdm(clip_paths, desc="Running frame ablation"):
        sample = torch.load(clip_path, weights_only=False)

        if predicted_class_name is not None and sample["predicted_class_name"] != predicted_class_name:
            continue

        dataset_index = video_path_to_index[sample["video_path"]]
        video_tensor, _ = val_dataset[dataset_index]  # [32, 3, 224, 224]
        video_tensor = video_tensor.unsqueeze(0).to(DEVICE)  # [1, 32, 3, 224, 224]

        target_class = sample["predicted_class"]

        drops = measure_frame_importance_by_occlusion(model, video_tensor, target_class)
        per_clip_drops.append(drops)

        clip_id = clip_path.stem
        result = {
            "confidence_drops": drops,  
            "video_path": sample["video_path"],
            "true_label": sample["true_label"],
            "true_class_name": sample["true_class_name"],
            "predicted_class": sample["predicted_class"],
            "predicted_class_name": sample["predicted_class_name"],
        }
        torch.save(result, output_dir / f"{clip_id}.pt")

    stacked = torch.stack(per_clip_drops) 
    mean_drops = stacked.mean(dim=0)

    return mean_drops, len(per_clip_drops)


if __name__ == "__main__":
    model = load_final_cnn_lstm_model()
    gradcam_dir = CNN_LSTM_XAI_RESULTS / "gradcam"
    mean_drops_all, n_all = run_test(model, gradcam_dir, DATASET_ROOT, RESULTS_DIR)
