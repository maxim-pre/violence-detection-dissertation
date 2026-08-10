import numpy as np 
import cv2
from pathlib import Path
from tqdm import tqdm
from src.config import DATASET_ROOT

def apply_gamma_correction(frame, gamma=0.67):
    """
    Apply gamma correction to an image.
    """

    # Convert pixel values to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    # Apply gamma correction
    frame = np.power(frame, gamma)
    # Convert back to orgianl range
    frame = (frame * 255).astype(np.uint8)
    return frame

def create_gamma_corrected_video(input_path, output_path, gamma=0.67):
    """
    Create a gamma corrected video from an input video.
    """

    # Open the input video
    cap = cv2.VideoCapture(input_path)

    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Apply gamma correction
        frame = apply_gamma_correction(frame, gamma)

        writer.write(frame)

    # Release resources
    cap.release()
    writer.release()

def build_gamma_dataset(dataset_root, output_root, gamma=0.67):
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    for video_path in tqdm(dataset_root.rglob("*.avi")):
        relative_path = video_path.relative_to(dataset_root)
        output_path = output_root / relative_path

        if output_path.exists():
            continue

        create_gamma_corrected_video(video_path, output_path, gamma=gamma)


if __name__ == "__main__":
    GAMMA_DATASET_ROOT = Path("/homes/mp2940/demo/datasets/rwf-2000/RWF-2000-gamma-067")

    build_gamma_dataset(DATASET_ROOT, GAMMA_DATASET_ROOT)
    




