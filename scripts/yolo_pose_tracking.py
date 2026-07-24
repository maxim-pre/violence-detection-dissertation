from pathlib import Path
import torch
import cv2
from ultralytics import YOLO
from tqdm import tqdm
from src.config import DATASET_ROOT


def extract_pose_tracking_data(
    video_path,
    model, 
    output_path=None, 
    tracker="bytetrack.yaml",
    imgsz=640,
    conf=0.25,
    device="cuda:2"

):
    
    results = model.track(source=video_path, stream=True, persist=True, imgsz=imgsz,
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

def save_pose_tracking_video(
    input_path,
    output_path,
    model_path="yolo26m-pose.pt",
    tracker="bytetrack.yaml",
    device="cuda:2",
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(input_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while True:
        success, frame = cap.read()
        if not success:
            break
        result = model.track(frame, persist=True, tracker=tracker, device=device, verbose=False,)[0]

        annotated_frame = result.plot()
        writer.write(annotated_frame)

    cap.release()
    writer.release()

    print(f"Saved video to {output_path}")


def save_sinlge_tracked_video(video_num, model_path, dataset):
    video_path = dataset.samples[video_num][0]
    video_path = str(video_path)
    output_path= f"outputs/video_{video_num}.mp4"
    save_pose_tracking_video(video_path, output_path, model_path=model_path)

def save_multiple_tracked_video(start, stop, model_path, dataset):
    for video_num in range(start, stop):
        save_sinlge_tracked_video(video_num, model_path, dataset)

def build_pose_dataset(dataset, output_root, model):

    for video_path, label in tqdm(dataset.samples):

        relative_path = video_path.relative_to(DATASET_ROOT)

        output_path = (
            output_root /
            relative_path.with_suffix(".pt")
        )

        extract_pose_tracking_data(video_path=video_path, output_path=output_path, model=model)
