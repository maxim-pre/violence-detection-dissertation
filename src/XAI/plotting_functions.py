import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


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