from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class RWF2000Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        num_frames=32,
        image_size=224,
    ):
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size

        self.label_map = {
            "NonFight": 0,
            "Fight": 1
        }

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

        for label_name, label_idx in self.label_map.items():
            folder = self.root_dir / self.split / label_name

            for video_path in folder.glob("*.avi"):
                samples.append((video_path, label_idx))

        return samples

    def __len__(self):
        return len(self.samples)

    def _sample_frame_indices(self, total_frames):
        # Sample frame indices uniformly across the video
        return np.linspace(
            0,
            total_frames - 1,
            self.num_frames,
            dtype=int
        )

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = self._sample_frame_indices(total_frames)

        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # opencv loads in BGR format, convert to RGB for pytorch
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frame = frame.astype(np.float32) / 255.0 # normalize pixel values to [0, 1]

            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1) # convert from HWC to CHW format for PyTorch

            frames.append(frame)

        cap.release()

        video_tensor = torch.stack(frames)

        return video_tensor

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        video = self._load_video(video_path)
        label = torch.tensor(label, dtype=torch.long)

        return video, label