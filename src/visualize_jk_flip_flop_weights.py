import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os
import glob
import argparse
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def visualize_jk_states(
    model_checkpoint_paths: List[str],
    output_dir: str,
    device: torch.device = torch.device("cpu"),
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Lists to store weights over epochs
    j_weights_x_history = []
    j_weights_h_history = []
    k_weights_x_history = []
    k_weights_h_history = []
    epochs = []

    for checkpoint_path in model_checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint["epoch"]

        j_weights_x = state_dict["cell.j_linear_x.weight"].cpu().numpy()
        j_weights_h = state_dict["cell.j_linear_h.weight"].cpu().numpy()
        k_weights_x = state_dict["cell.k_linear_x.weight"].cpu().numpy()
        k_weights_h = state_dict["cell.k_linear_h.weight"].cpu().numpy()

        j_weights_x_history.append(j_weights_x)
        j_weights_h_history.append(j_weights_h)
        k_weights_x_history.append(k_weights_x)
        k_weights_h_history.append(k_weights_h)
        epochs.append(epoch)

    for idx, epoch in enumerate(epochs):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f"JK Flip-Flop Internal Weights - Epoch {epoch}")

        im1 = ax1.imshow(j_weights_x_history[idx], cmap="coolwarm", aspect="auto")
        ax1.set_title("J Gate Input Weights")
        ax1.set_xlabel("Input Features")
        ax1.set_ylabel("Hidden Units")
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(j_weights_h_history[idx], cmap="coolwarm", aspect="auto")
        ax2.set_title("J Gate Hidden Weights")
        ax2.set_xlabel("Hidden Features")
        ax2.set_ylabel("Hidden Units")
        plt.colorbar(im2, ax=ax2)

        # Plot K gate weights
        im3 = ax3.imshow(k_weights_x_history[idx], cmap="coolwarm", aspect="auto")
        ax3.set_title("K Gate Input Weights")
        ax3.set_xlabel("Input Features")
        ax3.set_ylabel("Hidden Units")
        plt.colorbar(im3, ax=ax3)

        im4 = ax4.imshow(k_weights_h_history[idx], cmap="coolwarm", aspect="auto")
        ax4.set_title("K Gate Hidden Weights")
        ax4.set_xlabel("Hidden Features")
        ax4.set_ylabel("Hidden Units")
        plt.colorbar(im4, ax=ax4)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"jk_weights_epoch_{epoch}.png"))
        plt.close()

    # Create animation of weight changes
    create_weight_evolution_plots(
        j_weights_x_history,
        j_weights_h_history,
        k_weights_x_history,
        k_weights_h_history,
        epochs,
        output_dir,
    )


def create_weight_evolution_plots(
    j_weights_x_history: List[np.ndarray],
    j_weights_h_history: List[np.ndarray],
    k_weights_x_history: List[np.ndarray],
    k_weights_h_history: List[np.ndarray],
    epochs: List[int],
    output_dir: str,
) -> None:
    # Calculate statistics
    j_x_mean = [np.mean(w) for w in j_weights_x_history]
    j_x_std = [np.std(w) for w in j_weights_x_history]
    j_h_mean = [np.mean(w) for w in j_weights_h_history]
    j_h_std = [np.std(w) for w in j_weights_h_history]

    k_x_mean = [np.mean(w) for w in k_weights_x_history]
    k_x_std = [np.std(w) for w in k_weights_x_history]
    k_h_mean = [np.mean(w) for w in k_weights_h_history]
    k_h_std = [np.std(w) for w in k_weights_h_history]

    # Plot statistics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # J gate evolution
    ax1.plot(epochs, j_x_mean, label="J Input Weights Mean")
    ax1.fill_between(
        epochs,
        [m - s for m, s in zip(j_x_mean, j_x_std)],
        [m + s for m, s in zip(j_x_mean, j_x_std)],
        alpha=0.3,
    )
    ax1.plot(epochs, j_h_mean, label="J Hidden Weights Mean")
    ax1.fill_between(
        epochs,
        [m - s for m, s in zip(j_h_mean, j_h_std)],
        [m + s for m, s in zip(j_h_mean, j_h_std)],
        alpha=0.3,
    )
    ax1.set_title("J Gate Weights Evolution")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Weight Value")
    ax1.legend()
    ax1.grid(True)

    # K gate evolution
    ax2.plot(epochs, k_x_mean, label="K Input Weights Mean")
    ax2.fill_between(
        epochs,
        [m - s for m, s in zip(k_x_mean, k_x_std)],
        [m + s for m, s in zip(k_x_mean, k_x_std)],
        alpha=0.3,
    )
    ax2.plot(epochs, k_h_mean, label="K Hidden Weights Mean")
    ax2.fill_between(
        epochs,
        [m - s for m, s in zip(k_h_mean, k_h_std)],
        [m + s for m, s in zip(k_h_mean, k_h_std)],
        alpha=0.3,
    )
    ax2.set_title("K Gate Weights Evolution")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Weight Value")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weight_evolution.png"))
    plt.close()


def visualize_jk_states_video(
    model_checkpoint_paths: List[str],
    output_dir: str,
    fps: float = 1 / 0.3,  # 0.3 seconds per frame
    device: torch.device = torch.device("cpu"),
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Lists to store weights over epochs
    j_weights_x_history = []
    j_weights_h_history = []
    k_weights_x_history = []
    k_weights_h_history = []
    epochs = []

    # Load all checkpoints
    for checkpoint_path in sorted(
        model_checkpoint_paths, key=lambda x: int(x.split("_")[-1].split(".")[0])
    ):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint["epoch"]

        j_weights_x = state_dict["cell.j_linear_x.weight"].cpu().numpy()
        j_weights_h = state_dict["cell.j_linear_h.weight"].cpu().numpy()
        k_weights_x = state_dict["cell.k_linear_x.weight"].cpu().numpy()
        k_weights_h = state_dict["cell.k_linear_h.weight"].cpu().numpy()

        j_weights_x_history.append(j_weights_x)
        j_weights_h_history.append(j_weights_h)
        k_weights_x_history.append(k_weights_x)
        k_weights_h_history.append(k_weights_h)
        epochs.append(epoch)

    # Prepare video writer
    first_frame = create_frame(
        j_weights_x_history[0],
        j_weights_h_history[0],
        k_weights_x_history[0],
        k_weights_h_history[0],
        epochs[0],
    )
    height, width, layers = first_frame.shape

    video_path = os.path.join(output_dir, "jk_weights_evolution.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Create and write frames
    print("Creating video frames...")
    for idx, epoch in enumerate(epochs):
        frame = create_frame(
            j_weights_x_history[idx],
            j_weights_h_history[idx],
            k_weights_x_history[idx],
            k_weights_h_history[idx],
            epoch,
        )
        video_writer.write(frame)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(epochs)} frames")

    video_writer.release()
    print(f"Video saved to {video_path}")


def create_frame(
    j_weights_x: np.ndarray,
    j_weights_h: np.ndarray,
    k_weights_x: np.ndarray,
    k_weights_h: np.ndarray,
    epoch: int,
) -> np.ndarray:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f"JK Flip-Flop Internal Weights - Epoch {epoch}")

    # Plot J gate weights
    im1 = ax1.imshow(j_weights_x, cmap="coolwarm", aspect="auto")
    ax1.set_title("J Gate Input Weights")
    ax1.set_xlabel("Input Features")
    ax1.set_ylabel("Hidden Units")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(j_weights_h, cmap="coolwarm", aspect="auto")
    ax2.set_title("J Gate Hidden Weights")
    ax2.set_xlabel("Hidden Features")
    ax2.set_ylabel("Hidden Units")
    plt.colorbar(im2, ax=ax2)

    # Plot K gate weights
    im3 = ax3.imshow(k_weights_x, cmap="coolwarm", aspect="auto")
    ax3.set_title("K Gate Input Weights")
    ax3.set_xlabel("Input Features")
    ax3.set_ylabel("Hidden Units")
    plt.colorbar(im3, ax=ax3)

    im4 = ax4.imshow(k_weights_h, cmap="coolwarm", aspect="auto")
    ax4.set_title("K Gate Hidden Weights")
    ax4.set_xlabel("Hidden Features")
    ax4.set_ylabel("Hidden Units")
    plt.colorbar(im4, ax=ax4)

    plt.tight_layout()

    # Convert plot to image
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Convert to numpy array
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(int(height), int(width), 3)

    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    plt.close(fig)
    return image


def main(checkpoint_dir, output_dir, checkpoint_freq, min_checkpoint_to_apply_freq):
    all_checkpoint_paths = glob.glob(
        os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")
    )

    all_checkpoint_paths = sorted(
        all_checkpoint_paths, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    checkpoint_paths = [
        file
        for file in all_checkpoint_paths
        if int(file.split("_")[-1].split(".")[0]) < min_checkpoint_to_apply_freq
        or int(file.split("_")[-1].split(".")[0]) % int(checkpoint_freq) == 0
    ]

    visualize_jk_states(checkpoint_paths, output_dir)
    visualize_jk_states_video(all_checkpoint_paths, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Signal Generation benchmark with config"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_freq", type=int, default=500)
    parser.add_argument("--min_checkpoint_to_apply_freq", type=int, default=499)
    args = parser.parse_args()
    main(
        args.checkpoint_dir,
        args.output_dir,
        args.checkpoint_freq,
        args.min_checkpoint_to_apply_freq,
    )
