import torch
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from src.config import DATASET_ROOT
from scripts.common.get_device import get_available_device


def extract_pose_tracking_data(
    video_path,
    model, 
    output_path=None, 
    tracker="bytetrack.yaml",
    imgsz=640,
    conf=0.25,
    device="cuda:2"

):
    
    results = model.track(source=video_path, stream=True, persist=False, imgsz=imgsz,
                          tracker=tracker, conf=conf, device=device, verbose=False)

    frames = []
    original_video_shape = None

    frames_with_detections = 0
    frames_with_track_ids = 0
    frames_without_track_ids = 0

    for frame_index, result in enumerate(results):
        if original_video_shape is None: original_video_shape = tuple(result.orig_shape)

        boxes = result.boxes
        keypoints = result.keypoints

        frame_data = {
            "frame_index": frame_index,
            "people": [],
        }

        # 1. No people detected in the Frame
        if len(boxes) == 0:
            frames.append(frame_data)
            continue

        frames_with_detections += 1

        # ensure each detected person has both keypoints and a bounding box
        if len(keypoints) != len(boxes):
            raise RuntimeError(f"Frame {frame_index} contains {len(boxes)} boxes but {len(keypoints)} pose outputs.")
        
        # 2. people were detected but no id was assigned
        if boxes.id is None:
            frames_without_track_ids += 1
            frames.append(frame_data)
            continue

        track_ids = boxes.id.int().cpu()
        bounding_boxes = boxes.xyxy.cpu()
        box_confidences = boxes.conf.cpu()
        keypoint_coordinates = keypoints.xy.cpu()
        keypoint_confidences = keypoints.conf.cpu()

        # ensure each detected person has a tracking id
        if len(track_ids) != len(boxes):
            raise RuntimeError(f"Frame {frame_index}: found {len(boxes)} boxes but {len(track_ids)} tracking IDs.")

        frames_with_track_ids += 1

        for person_index in range(len(boxes)):
            frame_data["people"].append(
                {
                    "track_id": int(track_ids[person_index].item()),
                    "bbox": bounding_boxes[person_index].clone(),
                    "bbox_confidence": float(box_confidences[person_index].item()),
                    "keypoints": keypoint_coordinates[person_index].clone(), 
                    "keypoint_confidence": keypoint_confidences[person_index].clone()
                }
            )
        frames.append(frame_data)

    pose_data = {
        "video_path": str(video_path),
        "original_video_shape": original_video_shape, 
        "frames": frames, 
        "summary": {
            "frames_with_detections": frames_with_detections,
            "frames_without_detection": len(frames) - frames_with_detections,
            "frames_with_track_ids": frames_with_track_ids,
            "frames_without_track_ids": frames_without_track_ids,
            "detection_frame_coverage_percentage": frames_with_detections / len(frames),
            "tracking_frame_coverage_percentage": frames_with_track_ids / len(frames)
        }
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pose_data, output_path)

    return pose_data

def build_pose_dataset(
    dataset,
    dataset_root,
    output_root,
    model,
    device,
    tracker="bytetrack.yaml",
    imgsz=640,
    conf=0.25,
):
    dataset_root = dataset_root
    output_root = Path(output_root)

    failed_videos = []
    processed = 0
    skipped = 0

    for video_path, label in tqdm(
        dataset.samples,
        desc="Extracting pose data",
    ):
        video_path = Path(video_path)

        relative_path = video_path.relative_to(dataset_root)
        output_path = output_root / relative_path.with_suffix(".pt")

        if output_path.exists():
            skipped += 1
            continue

        try:
            extract_pose_tracking_data(
                video_path=video_path,
                model=model,
                output_path=output_path,
                tracker=tracker,
                imgsz=imgsz,
                conf=conf,
                device=device,
            )

            processed += 1

        except Exception as error:
            failed_videos.append(
                {
                    "video_path": str(video_path),
                    "output_path": str(output_path),
                    "error": repr(error),
                }
            )

            tqdm.write(
                f"Failed: {video_path.name} — {error}"
            )

    print("\nDataset build complete")
    print(f"Processed: {processed}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {len(failed_videos)}")

    return failed_videos

def pose_data_to_stgcn_tensor(pose_data, num_frames=150, num_keypoints=17, max_people=4, track_score_mode="total_confidence"):
    #convert pose data -> [3, 150, 17, 4]

    H, W = pose_data["original_video_shape"]

    # 1. Rank each track to determine which people to include in tensor
    track_confidence_sum = defaultdict(float) 
    track_joint_count = defaultdict(int)

    # loop over all people in the frame for every frame
    for frame in pose_data["frames"]:
        for person in frame["people"]:

            track_id = int(person["track_id"]) # get the track id
            confidences = person["keypoint_confidence"].float() # get the confidence values (array of length 17)

            track_confidence_sum[track_id] += confidences.sum().item() # score is based on the total of joint scores for every joint in every frame
            track_joint_count[track_id] += len(confidences) # (len 17)

    track_scores = {}

    for track_id in track_confidence_sum:

        if track_score_mode == "total_confidence":
            track_scores[track_id] = track_confidence_sum[track_id] # takes into account how many frames the person is iin

        elif track_score_mode == "mean_confidence":
            mean_confidence = track_confidence_sum[track_id] / track_joint_count[track_id] # average joint confidence across all frames
            total_tracked_frames = track_joint_count[track_id] / num_keypoints

            track_scores[track_id] = mean_confidence * (total_tracked_frames ** 0.5)
        else:
            raise ValueError(f"Unknown track_score_mode: {track_score_mode}")


    # select the strongest tracks according to max_people
    sorted_tracks = sorted(track_scores.items(), key=lambda item: item[1], reverse=True)
    selected_track_ids = [track_id for track_id, score in sorted_tracks[:max_people]]

    track_to_person_index = {}
    for person_index, track_id in enumerate(selected_track_ids):
        track_to_person_index[track_id] = person_index

    # construct the tensor
    tensor = torch.zeros(3, num_frames, num_keypoints, max_people, dtype=torch.float32)

    for frame in pose_data["frames"]:
        frame_index = frame["frame_index"]

        for person in frame["people"]:
            
            track_id = int(person["track_id"])
            if track_id not in track_to_person_index: continue # only add top four tracks

            person_index = track_to_person_index[track_id]
            keypoints = person["keypoints"]
            confidences = person["keypoint_confidence"]

            # normalise between 0-1
            x = keypoints[:, 0] / W
            y = keypoints[:, 1] / H

            tensor[0, frame_index, :, person_index] = x
            tensor[1, frame_index, :, person_index] = y
            tensor[2, frame_index, :, person_index] = confidences

    return tensor



