from pathlib import Path
import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses, model_name: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss per Epoch")
    plt.legend()

    loss_image_path = Path(output_dir) / f"{model_name}_loss.png"
    plt.savefig(loss_image_path)
    plt.close()


def plot_comparison(train_losses_list, val_losses_list, model_names, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for train_losses, val_losses, model_name in zip(
        train_losses_list, val_losses_list, model_names
    ):
        plt.plot(train_losses, label=f"{model_name} Training Loss")
        plt.plot(val_losses, label=f"{model_name} Validation Loss", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Comparison Across Models")
    plt.legend()

    comparison_image_path = Path(output_dir) / "model_loss_comparison.png"
    plt.savefig(comparison_image_path)
    plt.close()
