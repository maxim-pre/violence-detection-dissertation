import matplotlib.pyplot as plt


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