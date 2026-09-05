import csv
import gc
import json
import torch
from pathlib import Path
from tqdm import tqdm

from src.rwf2000 import RWF2000Dataset
from src.cnn_lstm.cnn_lstm_v3 import CNNLSTMV3
from src.config import CHECKPOINT_DIR, DATASET_ROOT, CNN_LSTM_XAI_RESULTS
from src.XAI.gradcam import GradCAM
from src.XAI.full_gradcam import FullGradCAM
from src.XAI.smooth_gradcam import SmoothGradCAM
from src.XAI.rise import RISE
from src.XAI.insertion_deletion_auc import AUC_TEST
from src.XAI.RIS import RIS_TEST
from scripts.common.get_device import get_available_device

CHECKPOINT_PATH = CHECKPOINT_DIR / "CNN_LSTM_V4" / "final_systematic_search_2" / "h64_d0.35_wd2e-05_ls0.1" / "best_model.pt"
XAI_RESULTS_ROOT = CNN_LSTM_XAI_RESULTS  
RISE_NUM_MASKS = 16000                  
RIS_NUM_PERTURBATIONS = 20
RIS_CLIP_LIST_PATH = Path("/homes/mp2940/violence-detection-dissertation/scripts/common/XAI_scripts/rise_evaluation_clips.json") # only doing 20 clips for perturbation methods
AUC_NUM_STEPS = 100
MAX_CLIPS = None

CLASS_NAMES = {0: "NonFight", 1: "Fight"}
TECHNIQUE_NAMES = ["gradcam", "multilayer_gradcam", "smoothgrad_cam", "rise"]


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
        pooling_mode=config.get("pooling_mode", "double"),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def squeeze_saliency_map(saliency_map): # ensure the batch dimension is removed

    if saliency_map.dim() == 4 and saliency_map.shape[0] == 1:
        saliency_map = saliency_map.squeeze(0)
    return saliency_map


def load_saved_saliency_map(xai_results_root, technique, clip_id, device):
    path = xai_results_root / technique / f"{clip_id}.pt"
    saved = torch.load(path, map_location=device, weights_only=False)
    return squeeze_saliency_map(saved["saliency_map"]), saved["predicted_class"]


def run_auc_evaluation(auc_test, input_tensor, saliency_map, target_class):
    insertion_fractions, insertion_scores, insertion_auc = auc_test.insertion(input_tensor, saliency_map, target_class=target_class)
    deletion_fractions, deletion_scores, deletion_auc = auc_test.deletion(input_tensor, saliency_map, target_class=target_class)

    return {
        "insertion_fractions": insertion_fractions.cpu(),
        "insertion_scores": insertion_scores.cpu(),
        "insertion_auc": insertion_auc,
        "deletion_fractions": deletion_fractions.cpu(),
        "deletion_scores": deletion_scores.cpu(),
        "deletion_auc": deletion_auc,
    }


def make_explain_fn(technique, target_class):
    def explain_fn(input_tensor):
        saliency_map = technique.generate_heatmap(input_tensor, target_class=target_class)
        return squeeze_saliency_map(saliency_map)
    return explain_fn


def evaluate(model, val_dataset, xai_techniques, xai_results_root, ris_clip_ids,
             auc_test, ris_test, output_root, max_clips=None):

    output_root.mkdir(parents=True, exist_ok=True)
    curves_root = output_root / "auc_curves"
    for name in xai_techniques:
        (curves_root / name).mkdir(parents=True, exist_ok=True)

    auc_path = output_root / "auc_results.csv"
    ris_path = output_root / "ris_results.csv"

    write_auc_header = not auc_path.exists()
    write_ris_header = not ris_path.exists()

    # resume support: load what's already been done if any
    done_auc = set()
    if auc_path.exists():
        with open(auc_path) as f:
            for row in csv.DictReader(f):
                done_auc.add((row["clip_id"], row["technique"]))

    done_ris = set()
    if ris_path.exists():
        with open(ris_path) as f:
            for row in csv.DictReader(f):
                done_ris.add((row["clip_id"], row["technique"]))

    num_clips = len(val_dataset) if max_clips is None else min(max_clips, len(val_dataset))

    with open(auc_path, "a", newline="") as auc_file, \
         open(ris_path, "a", newline="") as ris_file:

        auc_writer = csv.writer(auc_file)
        if write_auc_header: auc_writer.writerow(["clip_id", "technique", "insertion_auc", "deletion_auc"])

        ris_writer = csv.writer(ris_file)
        if write_ris_header: ris_writer.writerow(["clip_id", "technique", "ris_value", "num_perturbations"])

        for clip_index in tqdm(range(num_clips), desc="CNN-LSTM quantitative eval"):
            video_path, _ = val_dataset.samples[clip_index]
            video_path = Path(video_path)
            clip_id = f"{video_path.parent.name}_{video_path.stem}"

            video, label = val_dataset[clip_index]
            input_tensor = video.unsqueeze(0).to(next(model.parameters()).device)

            for technique_name in TECHNIQUE_NAMES:
                saliency_map, predicted_class = load_saved_saliency_map(xai_results_root, technique_name, clip_id, input_tensor.device)

                if (clip_id, technique_name) not in done_auc:
                    try:
                        auc_result = run_auc_evaluation(auc_test, input_tensor, saliency_map, predicted_class)
                    except torch.cuda.OutOfMemoryError as error:
                        print(f"OOM during AUC for {clip_id}/{technique_name}: {error}")
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue

                    auc_writer.writerow([clip_id, technique_name, auc_result["insertion_auc"], auc_result["deletion_auc"]])
                    torch.save(auc_result, curves_root / technique_name / f"{clip_id}.pt")
                    done_auc.add((clip_id, technique_name))

                # only run RIS on the subset for RISE
                run_ris_here = technique_name != "rise" or clip_id in ris_clip_ids

                if run_ris_here and (clip_id, technique_name) not in done_ris:
                    try:
                        explain_fn = make_explain_fn(xai_techniques[technique_name], predicted_class)
                        ris_value = ris_test.evaluate(input_tensor, saliency_map, explain_fn, num_perturbations=RIS_NUM_PERTURBATIONS)
                    except torch.cuda.OutOfMemoryError as error:
                        print(f"OOM during RIS for {clip_id}/{technique_name}: {error}")
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue

                    ris_writer.writerow([clip_id, technique_name, ris_value, RIS_NUM_PERTURBATIONS])
                    done_ris.add((clip_id, technique_name))

            auc_file.flush()
            ris_file.flush()


if __name__ == "__main__":
    device = get_available_device()
    model = load_model(CHECKPOINT_PATH, device)

    val_dataset = RWF2000Dataset(root_dir=DATASET_ROOT, split="val", input_mode="diff")

    xai_techniques = {
        "gradcam": GradCAM(model, target_layer=model.cnn[-1], normalisation_mode="per_video"),
        "multilayer_gradcam": FullGradCAM(model, target_layers=[model.cnn[i] for i in range(19)], normalisation_mode="per_video"),
        "smoothgrad_cam": SmoothGradCAM(model, target_layer=model.cnn[-1], normalisation_mode="per_video"),
        "rise": RISE(model, num_masks=RISE_NUM_MASKS, normalisation_mode="per_video"),
    }

    auc_test = AUC_TEST(model, data_type="image", num_steps=AUC_NUM_STEPS)
    ris_test = RIS_TEST(model, data_type="image", noise_std=0.01, noise_mode="per_video", max_attempts=5000)

    with open(RIS_CLIP_LIST_PATH) as f:
        ris_clip_ids = set(json.load(f))

    evaluate(
        model=model,
        val_dataset=val_dataset,
        xai_techniques=xai_techniques,
        xai_results_root=XAI_RESULTS_ROOT,
        ris_clip_ids=ris_clip_ids,
        auc_test=auc_test,
        ris_test=ris_test,
        output_root=XAI_RESULTS_ROOT,
        max_clips=MAX_CLIPS,
    )