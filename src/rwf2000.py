from pathlib import Path
import cv2
import torch
from torchvision.transforms import Normalize
from torch.utils.data import Dataset
import numpy as np
import random


class RWF2000Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        num_frames=32,
        image_size=224,
        crop_resize_size=320,
        augment=False
    ):
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size,
        self.crop_resize_size = crop_resize_size
        self.augment = augment

        self.label_map = {
            "NonFight": 0,
            "Fight": 1
        }

        self.samples = self._load_samples()

        # image normalisation for mobile net
        self.normalise = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _load_samples(self):
        samples = []

        for label_name, label_idx in self.label_map.items():
            folder = self.root_dir / self.split / label_name

            for video_path in folder.glob("*.avi"):
                samples.append((video_path, label_idx))

        return samples

    def _get_augmentation_params(self):

        if not self.augment:
            return {
                "flip": False,
                "brightness": 1.0,
                "contrast": 1.0,
                "crop_scale": 1.0,
                "crop_x": 0.0,
                "crop_y": 0.0,
            }

        crop_scale = random.uniform(0.85, 1.0)

        return {
            "flip": random.random() < 0.5,
            "brightness": random.uniform(0.85, 1.15),
            "contrast": random.uniform(0.85, 1.15),
            "crop_scale": crop_scale,
            "crop_x": random.uniform(0.0, 1.0),
            "crop_y": random.uniform(0.0, 1.0),
        }

    def _apply_augmentation(self, frame, aug):

        frame = cv2.resize(frame, (self.crop_resize_size, self.crop_resize_size))

        h, w, _ = frame.shape

        # Random crop, same crop position for all frames in a clip
        if aug["crop_scale"] < 1.0:
            crop_h = int(h * aug["crop_scale"])
            crop_w = int(w * aug["crop_scale"])

            max_y = h - crop_h
            max_x = w - crop_w

            y1 = int(max_y * aug["crop_y"])
            x1 = int(max_x * aug["crop_x"])

            frame = frame[y1:y1 + crop_h, x1:x1 + crop_w]

        # Resize back to model input size
        frame = cv2.resize(frame, (self.image_size, self.image_size))

        # Horizontal flip
        if aug["flip"]:
            frame = cv2.flip(frame, 1)

        # Brightness + contrast
        frame = frame.astype(np.float32)
        frame = frame * aug["contrast"]
        frame = frame + (aug["brightness"] - 1.0) * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        return frame

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
        aug_params = self._get_augmentation_params()

        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # opencv loads in BGR format, convert to RGB for pytorch

            if self.augment:
                frame = self._apply_augmentation(frame, aug_params)
            else:
                frame = cv2.resize(frame, (self.image_size, self.image_size))

            frame = frame.astype(np.float32) / 255.0 # normalize pixel values to [0, 1]
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1) # convert from HWC to CHW format for PyTorch
            frame = self.normalise(frame)

            frames.append(frame)

        cap.release()

        video_tensor = torch.stack(frames)

        return video_tensor

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        video = self._load_video(video_path)
        label = torch.tensor(label, dtype=torch.long)

        return video, label