import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
import cv2
import numpy as np
from pathlib import Path

def plot_gradcam_over_rgb(heatmaps, rgb_frames, label, prediction, alpha=0.45):
    class_labels = {0: "Non-Violent", 1: "Violent"}

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
        f"Grad-CAM\n"
        f"Prediction: {class_labels[prediction]} | "
        f"Truth: {class_labels[label]}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()


def plot_single_frame_gradcam(heatmaps, rgb_frames, frame_index, label, prediction, alpha=0.45):
    class_labels = {0: "Non-Violent", 1: "Violent"}

    rgb_frame = rgb_frames[frame_index + 1]
    rgb_frame = rgb_frame.permute(1, 2, 0).cpu()
    heatmap = heatmaps[frame_index].cpu()

    plt.figure(figsize=(12, 12))
    plt.imshow(rgb_frame)
    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=alpha,
        vmin=0,
        vmax=1,
    )
    plt.title(
        f"Grad-CAM\n"
        f"Prediction: {class_labels[prediction]} | "
        f"Truth: {class_labels[label]}",
        fontsize=16,
    )
    plt.axis("off") 
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

def plot_joint_occlusion(
    rgb_frames,
    pose_tensor,
    joint_importance,
    frame_index,
):
    """
    Overlay joint-occlusion importance for all tracked people.

    rgb_frames:      [T, 3, H, W]
    pose_tensor:     [1, 3, T, V, M]
    joint_importance:[T, V, M]
    """

    # Convert the RGB frame from [C, H, W] to [H, W, C].
    frame = rgb_frames[frame_index].permute(1, 2, 0).cpu()
    H, W, _ = frame.shape

    pose = pose_tensor[0].detach().cpu()
    num_people = pose.shape[-1]

    # Use one colour scale for the whole video.
    max_score = max(joint_importance.abs().max().item(), 1e-8)
    colour_norm = Normalize(vmin=-max_score, vmax=max_score)
    colour_map = plt.get_cmap("coolwarm")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(frame)

    for person_index in range(num_people):
        x = pose[0, frame_index, :, person_index] * W
        y = pose[1, frame_index, :, person_index] * H
        confidence = pose[2, frame_index, :, person_index]

        # Skip padded person slots containing no pose data.
        if confidence.sum().item() == 0:
            continue

        scores = joint_importance[frame_index, :, person_index].detach().cpu()

        # Colour each connection using the mean score of its endpoint joints.
        for joint_a, joint_b in SKELETON_EDGES:
            edge_score = (scores[joint_a] + scores[joint_b]) / 2

            ax.plot(
                [x[joint_a], x[joint_b]],
                [y[joint_a], y[joint_b]],
                color=colour_map(colour_norm(edge_score.item())),
                linewidth=3,
            )

        # Draw every joint supplied to the model.
        for joint_index in range(len(x)):
            ax.scatter(
                x[joint_index],
                y[joint_index],
                color=colour_map(
                    colour_norm(scores[joint_index].item())
                ),
                s=80,
                edgecolor="black",
                zorder=2,
            )

    ax.set_title(f"Frame {frame_index}")
    ax.axis("off")

    plt.show()


def importance_to_bgr(score, max_score, colour_map):
    """
    Map an importance score from [-max_score, max_score]
    to an OpenCV BGR colour.
    """
    normalised = (score + max_score) / (2 * max_score)
    normalised = np.clip(normalised, 0, 1)

    red, green, blue, _ = colour_map(normalised)

    return (
        int(blue * 255),
        int(green * 255),
        int(red * 255),
    )


def save_joint_occlusion_video(
    rgb_frames,
    pose_tensor,
    joint_importance,
    output_path,
    fps=30,
):
    """
    Save an MP4 containing the joint-importance overlay.

    rgb_frames:       [T, 3, H, W]
    pose_tensor:      [1, 3, T, V, M]
    joint_importance: [T, V, M]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_frames = rgb_frames.detach().cpu()
    pose = pose_tensor[0].detach().cpu()
    scores = joint_importance.detach().cpu()

    num_frames, _, height, width = rgb_frames.shape
    num_people = pose.shape[-1]

    # Keep colours comparable across all frames.
    max_score = max(scores.abs().max().item(), 1e-8)
    colour_map = plt.get_cmap("coolwarm")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame_index in range(num_frames):
        # Convert [C, H, W] RGB tensor to an OpenCV BGR image.
        frame = rgb_frames[frame_index].permute(1, 2, 0).numpy()
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for person_index in range(num_people):
            x = pose[0, frame_index, :, person_index] * width
            y = pose[1, frame_index, :, person_index] * height
            confidence = pose[2, frame_index, :, person_index]

            # Confidence 0 represents a missing joint.
            valid_joints = confidence > 0

            if not valid_joints.any():
                continue

            person_scores = scores[
                frame_index,
                :,
                person_index,
            ]

            # Draw skeleton connections.
            for joint_a, joint_b in SKELETON_EDGES:
                if not (
                    valid_joints[joint_a]
                    and valid_joints[joint_b]
                ):
                    continue

                edge_score = (
                    person_scores[joint_a]
                    + person_scores[joint_b]
                ) / 2

                colour = importance_to_bgr(
                    edge_score.item(),
                    max_score,
                    colour_map,
                )

                point_a = (
                    int(x[joint_a].item()),
                    int(y[joint_a].item()),
                )

                point_b = (
                    int(x[joint_b].item()),
                    int(y[joint_b].item()),
                )

                cv2.line(
                    frame,
                    point_a,
                    point_b,
                    colour,
                    thickness=3,
                )

            # Draw joints.
            for joint_index in range(len(x)):
                if not valid_joints[joint_index]:
                    continue

                colour = importance_to_bgr(
                    person_scores[joint_index].item(),
                    max_score,
                    colour_map,
                )

                point = (
                    int(x[joint_index].item()),
                    int(y[joint_index].item()),
                )

                cv2.circle(
                    frame,
                    point,
                    radius=5,
                    color=colour,
                    thickness=-1,
                )

                cv2.circle(
                    frame,
                    point,
                    radius=5,
                    color=(0, 0, 0),
                    thickness=1,
                )

        cv2.putText(
            frame,
            f"Frame {frame_index}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    writer.release()

    print(f"Saved video to: {output_path}")




