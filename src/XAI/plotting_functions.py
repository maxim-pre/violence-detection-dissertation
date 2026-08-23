import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
import cv2
import numpy as np
from pathlib import Path

def _squeeze_batch_dim(heatmaps):
    if heatmaps.dim() == 4 and heatmaps.shape[0] == 1:
        heatmaps = heatmaps.squeeze(0)
    return heatmaps
 
 
def plot_heatmap_over_rgb(heatmaps, rgb_frames, label, prediction, alpha=0.45, technique_name="gradcam"):
    class_labels = {0: "Non-Violent", 1: "Violent"}
 
    heatmaps = _squeeze_batch_dim(heatmaps)
    num_frames = heatmaps.shape[0]
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
 
    for i, ax in enumerate(axes.flatten()):
        if i >= num_frames:
            ax.axis("off")
            continue
 
        rgb_frame = rgb_frames[i + 1]
 
        # [C, H, W] -> [H, W, C]
        rgb_frame = rgb_frame.permute(1, 2, 0).cpu()
 
        heatmap = heatmaps[i].cpu()
 
        ax.imshow(rgb_frame)
        ax.imshow(
            heatmap,
            cmap="jet",
            alpha=alpha,
            vmin=0,
            vmax=1,
        )
 
        ax.set_title(f"Frame {i + 1}")
        ax.axis("off")
    plt.suptitle(
        f"{technique_name}\n"
        f"Prediction: {class_labels[prediction]} | "
        f"Truth: {class_labels[label]}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()
 
def animate_saliency_over_rgb(
    heatmaps,
    rgb_frames,
    label,
    prediction,
    alpha=0.45,
    interval=150,
):
    """
    heatmaps: [num_frames, H, W]
    rgb_frames: [num_frames + 1, 3, H, W]
    interval: Delay between frames in milliseconds.
    """

    class_names = {
        0: "NonFight",
        1: "Fight",
    }

    heatmaps = heatmaps.detach().cpu()
    rgb_frames = rgb_frames.detach().cpu()

    num_frames = heatmaps.shape[0]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Heatmap 0 corresponds to RGB frame 1.
    first_rgb = rgb_frames[1].permute(1, 2, 0)
    first_heatmap = heatmaps[0]

    rgb_display = ax.imshow(first_rgb)

    heatmap_display = ax.imshow(
        first_heatmap,
        cmap="jet",
        alpha=alpha,
        vmin=0,
        vmax=1,
    )

    title = ax.set_title(
        f"Frame 1/{num_frames}\n"
        f"Prediction: {class_names[prediction]} | "
        f"Ground Truth: {class_names[label]}"
    )

    ax.axis("off")

    def update(frame_index):
        # Use frame_index + 1 because each heatmap explains a frame difference.
        rgb_frame = rgb_frames[frame_index + 1].permute(1, 2, 0)
        heatmap = heatmaps[frame_index]

        rgb_display.set_data(rgb_frame)
        heatmap_display.set_data(heatmap)

        title.set_text(
            f"Frame {frame_index + 1}/{num_frames}\n"
            f"Prediction: {class_names[prediction]} | "
            f"Ground Truth: {class_names[label]}"
        )

        return rgb_display, heatmap_display, title

    animation = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval,
        blit=False,
        repeat=True,
    )

    plt.close(fig)

    return animation

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

def plot_joint_importance_grid(rgb_frames, pose_tensor, joint_importance, label, prediction, num_frames_to_show=32):
    """
    Grid of evenly-spaced frames with skeleton overlays coloured by joint
    importance, styled to match plot_gradcam_over_rgb's layout for the
    CNN-LSTM so the two models' figures look consistent.
 
    rgb_frames:       [T, 3, H, W]
    pose_tensor:      [1, 3, T, V, M]
    joint_importance: [T, V, M] - assumed non-negative and normalised to
                       [0, 1], matching every technique in this work after
                       ReLU + per_video min-max normalisation.
    """
    class_names = {0: "NonFight", 1: "Fight"}
 
    T = joint_importance.shape[0]
    frame_indices = np.linspace(0, T - 1, num_frames_to_show).astype(int)
 
    pose = pose_tensor[0].detach().cpu()
    num_people = pose.shape[-1]
    scores_all = joint_importance.detach().cpu()
 
    # colour scale shared across every frame in the grid, so importance is
    # comparable frame-to-frame rather than each subplot rescaling itself
    max_score = max(scores_all.max().item(), 1e-8)
    colour_map = plt.get_cmap("jet")
    colour_norm = Normalize(vmin=0, vmax=max_score)
 
    fig, axes = plt.subplots(4, 8, figsize=(20, 11))
 
    for i, ax in enumerate(axes.flatten()):
        if i >= len(frame_indices):
            ax.axis("off")
            continue
 
        frame_index = int(frame_indices[i])
 
        frame = rgb_frames[frame_index].permute(1, 2, 0).cpu()
        H, W, _ = frame.shape
        ax.imshow(frame)
 
        for person_index in range(num_people):
            x = pose[0, frame_index, :, person_index] * W
            y = pose[1, frame_index, :, person_index] * H
            confidence = pose[2, frame_index, :, person_index]
 
            # skip padded person slots containing no pose data
            if confidence.sum().item() == 0:
                continue
 
            scores = scores_all[frame_index, :, person_index]
 
            # only draw an edge if both endpoint joints were actually detected -
            # otherwise a missing joint's (0, 0) padded coordinate gets drawn
            # as a real point, producing a spurious line to the frame corner
            for joint_a, joint_b in SKELETON_EDGES:
                if confidence[joint_a] == 0 or confidence[joint_b] == 0:
                    continue
 
                edge_score = (scores[joint_a] + scores[joint_b]) / 2
                ax.plot(
                    [x[joint_a], x[joint_b]],
                    [y[joint_a], y[joint_b]],
                    color=colour_map(colour_norm(edge_score.item())),
                    linewidth=3,        # was 2 - thicker for visibility
                    alpha=0.9,
                    solid_capstyle="round",
                )
 
            for joint_index in range(len(x)):
                if confidence[joint_index] == 0:
                    continue
 
                ax.scatter(
                    x[joint_index],
                    y[joint_index],
                    color=colour_map(colour_norm(scores[joint_index].item())),
                    s=45,               # was 25 - larger markers
                    edgecolor="white",  # was black - reads better on both dark and light backgrounds
                    linewidth=1,
                    zorder=3,
                )
 
        ax.set_title(f"Frame {frame_index}", fontsize=9)
        ax.axis("off")
 
    plt.suptitle(
        f"Joint Importance\n"
        f"Prediction: {class_names[prediction]} | "
        f"Truth: {class_names[label]}",
        fontsize=16,
    )
 
    # single shared colorbar for the whole grid, rather than per-subplot -
    # without this there is no way to tell what a given colour actually
    # means, or whether a uniform-looking grid reflects genuinely uniform
    # importance or just poor contrast
    scalar_mappable = plt.cm.ScalarMappable(norm=colour_norm, cmap=colour_map)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=axes, fraction=0.02, pad=0.02, shrink=0.8)
    colorbar.set_label("Joint / edge importance")
 
    plt.show()




