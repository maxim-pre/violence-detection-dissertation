import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
import cv2
import numpy as np
from pathlib import Path

# skeleton plotting functions

SKELETON_EDGES = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
]


def plot_saliency(rgb_frame, saliency_frame, ax=None, is_skeleton=False, pose_frame=None,
                   title=None, max_alpha=0.75, cmap="jet",
                   discretise=False, thresholds=(0.33, 0.66), discrete_colours=("blue", "yellow", "red")):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    rgb = rgb_frame.permute(1, 2, 0).cpu().numpy()
    H, W = rgb.shape[:2]
    ax.imshow(rgb)

    if discretise and len(discrete_colours) != len(thresholds) + 1:
        raise ValueError("discrete_colours must have exactly len(thresholds) + 1 entries")

    def get_colour(normalised_value, colour_map):
        if not discretise:
            return colour_map(normalised_value)

        for bin_index, threshold in enumerate(thresholds):
            if normalised_value < threshold:
                return discrete_colours[bin_index]
        return discrete_colours[-1]

    if not is_skeleton:
        sal = saliency_frame.cpu().numpy()
        colour_map = plt.get_cmap(cmap)
        coloured = colour_map(sal)
        coloured[..., 3] = sal * max_alpha
        ax.imshow(coloured, extent=(0, W, H, 0))

    else:
        colour_map = plt.get_cmap(cmap)
        colour_norm = Normalize(vmin=0, vmax=1.0)
        num_people = pose_frame.shape[-1]

        for person_index in range(num_people):
            confidence = pose_frame[2, :, person_index]
            x = pose_frame[0, :, person_index] * W
            y = pose_frame[1, :, person_index] * H
            scores = saliency_frame[:, person_index]

            if confidence.sum() == 0:
                continue

            for joint_a, joint_b in SKELETON_EDGES:
                if confidence[joint_a] == 0 or confidence[joint_b] == 0:
                    continue
                edge_score = (scores[joint_a] + scores[joint_b]) / 2
                edge_colour = get_colour(colour_norm(edge_score.item()), colour_map)
                ax.plot(
                    [x[joint_a], x[joint_b]], [y[joint_a], y[joint_b]],
                    color=edge_colour,
                    linewidth=1, alpha=0.9, solid_capstyle="round",
                )

            for joint_index in range(len(x)):
                if confidence[joint_index] == 0:
                    continue
                joint_colour = get_colour(colour_norm(scores[joint_index].item()), colour_map)
                ax.scatter(
                    x[joint_index], y[joint_index],
                    color=joint_colour,
                    s=15, edgecolor="white", linewidth=0.2, zorder=3,
                )

    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_saliency_grid(rgb_frames, saliency_map, is_skeleton=False, pose_tensor=None,
                        label=None, prediction=None, num_frames_to_show=32, **kwargs):

    class_names = {0: "NonFight", 1: "Fight"}
    T = saliency_map.shape[0]
    frame_indices = np.linspace(0, T - 1, num_frames_to_show).astype(int)

    ncols = 8
    nrows = -(-num_frames_to_show // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes = axes.flatten()

    for ax, frame_index in zip(axes, frame_indices):
        frame_index = int(frame_index)
        pose_frame = pose_tensor[0, :, frame_index] if is_skeleton else None
        plot_saliency(
            rgb_frames[frame_index], saliency_map[frame_index], ax=ax,
            is_skeleton=is_skeleton, pose_frame=pose_frame,
            title=f"Frame {frame_index}", **kwargs,
        )

    for ax in axes[len(frame_indices):]:
        ax.axis("off")

    if label is not None and prediction is not None:
        plt.suptitle(f"Prediction: {class_names[prediction]} | Truth: {class_names[label]}", fontsize=14)
    plt.tight_layout()
    plt.show()



