from pathlib import Path
import cv2
import torch
from torchvision.transforms import Normalize
from torch.utils.data import Dataset
from src.config import DEFAULT_AUGMENTATION_PARAMS, DEFAULT_POSE_AUGMENTATION_PARAMS
from src.Skeleton_model.yolo_pose_tracking import pose_data_to_stgcn_tensor
import math
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
        augment=False,
        augmentation_params=DEFAULT_AUGMENTATION_PARAMS,
        input_mode="rgb",
        return_rgb_frames=False
    ):
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        self.crop_resize_size = crop_resize_size
        self.augment = augment
        self.augmentation_params = augmentation_params
        self.input_mode = input_mode
        self.return_rgb_frames = return_rgb_frames

        if self.input_mode not in ["rgb", "diff"]:
            raise ValueError("Invalid input_mode. Must be 'rgb' or 'diff'.")

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

        crop_scale = random.uniform(self.augmentation_params["crop_scale_range"][0], self.augmentation_params["crop_scale_range"][1])

        return {
            "flip": random.random() < self.augmentation_params["flip_prob"],
            "brightness": random.uniform(self.augmentation_params["brightness_range"][0], self.augmentation_params["brightness_range"][1]),
            "contrast": random.uniform(self.augmentation_params["contrast_range"][0], self.augmentation_params["contrast_range"][1]),
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

    def _sample_frame_indices(self, total_frames, frames_to_sample):
        # Sample frame indices uniformly across the video
        return np.linspace(
            0,
            total_frames - 1,
            frames_to_sample,
            dtype=int
        )

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_sample = self.num_frames + 1 if self.input_mode == "diff" else self.num_frames

        frame_indices = self._sample_frame_indices(total_frames, frames_to_sample)
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

            frames.append(frame)

        cap.release()

        if self.return_rgb_frames:
            rbg_frames = []

            for frame in frames:
                frame = frame.astype(np.float32) / 255.0 # normalize pixel values to [0, 1]
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1) # convert from HWC to CHW format for PyTorch
                rbg_frames.append(frame)

        if self.input_mode == "rgb":
            processed_frames = []

            for frame in frames:
                frame = frame.astype(np.float32) / 255.0 # normalize pixel values to [0, 1]
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1) # convert from HWC to CHW format for PyTorch
                frame = self.normalise(frame)
                processed_frames.append(frame)
        
        elif self.input_mode == "diff":
            processed_frames = []

            for i in range(len(frames) -1):
                diff = frames[i+1].astype(np.float32) - frames[i].astype(np.float32)

                # Scale differences from [-255, 255] to [-1, 1]
                diff = diff / 255.0

                diff = torch.from_numpy(diff)
                diff = diff.permute(2, 0, 1)

                processed_frames.append(diff)

        video_tensor = torch.stack(processed_frames) 

        if self.return_rgb_frames:
            rbg_tensor = torch.stack(rbg_frames)
            return video_tensor, rbg_tensor

        return video_tensor

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        if self.return_rgb_frames:
            video, rgb_frames = self._load_video(video_path)
            label = torch.tensor(label, dtype=torch.long)

            return video, label, rgb_frames

        video = self._load_video(video_path)
        label = torch.tensor(label, dtype=torch.long)

        return video, label

class RWF2000PoseDataset(Dataset):

    def __init__(self, root_dir, num_frames=150, num_keypoints=17, max_people=2, split="train", augment=False, augment_params=DEFAULT_POSE_AUGMENTATION_PARAMS):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.num_keypoints = num_keypoints
        self.max_people = max_people
        self.split = split
        self.augment = augment
        self.augment_params = augment_params
        self.label_map = {
                "NonFight": 0,
                "Fight": 1
            }

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

        for label_name, label_idx in self.label_map.items():
            folder = self.root_dir / self.split / label_name

            for video_path in folder.glob("*.pt"):
                samples.append((video_path, label_idx))

        return samples

    def __len__(self):
        return len(self.samples)

    def _augment_pose(self, tensor):
        """
        tensor shape: [3, T, V, M]
        channels: x, y, confidence
        """
        tensor = tensor.clone()

        return tensor

    def __getitem__(self, index):
        pose_path, label = self.samples[index]

        pose_data = torch.load(pose_path, weights_only=False)
        tensor = pose_data_to_stgcn_tensor(
            pose_data=pose_data,
            num_frames=self.num_frames,
            num_keypoints=self.num_keypoints,
            max_people=self.max_people,
        )

        if self.augment:
            tensor = self._augment_pose(tensor)

        label = torch.tensor(label, dtype=torch.long)

        return tensor, label
        
