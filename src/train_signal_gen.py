import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List
import time
import wandb
import argparse
from matplotlib import pyplot as plt
import os
from models.recurrent import SignalRNN, SignalLSTM, SignalGRU
from common.utils import seed_models, load_config
from common.plotting import plot_losses, plot_comparison
from common.logging import conditional_log

seed_models()


class SignalGenerationDataset(Dataset):
    def __init__(
        self,
        num_signals_per_class: int,
        seq_length: int,
        num_classes: int,
        noise_std: float,
        device: torch.device,
    ):
        self.signals = []
        self.labels = []

        # Parameters for signal generation
        for class_idx in range(num_classes):

            # Random parameters for each signal
            amplitude = torch.rand(1).item() * 0.5 + 0.5  # Range [0.5, 1.0]
            frequency = torch.rand(1).item() * 2 + 1  # Range [1, 3] Hz
            phase = torch.rand(1).item() * 2 * np.pi  # Range [0, 2π]

            # Time vector
            t = torch.linspace(0, 10, seq_length)

            # Generate base signal based on class
            if class_idx == 0:
                # Class 0: Simple sinusoid
                signal = amplitude * torch.sin(2 * np.pi * frequency * t + phase)
            else:
                # Class 1: Sum of two sinusoids
                freq2 = frequency * 2
                signal = amplitude * (
                    torch.sin(2 * np.pi * frequency * t + phase)
                    + 0.5 * torch.sin(2 * np.pi * freq2 * t + phase)
                )

            # Create one-hot encoded label
            label = torch.zeros(num_classes)
            label[class_idx] = 1.0

            for _ in range(num_signals_per_class):
                # Add Gaussian noise
                cur_noise = torch.randn_like(signal) * noise_std
                cur_signal = signal + cur_noise

                self.signals.append(cur_signal)
                self.labels.append(label)

        self.signals = (
            torch.stack(self.signals).unsqueeze(-1).to(device)
        )  # [num_signals_per_class * seq_length, seq_length, 1]
        self.labels = torch.stack(self.labels).to(
            device
        )  # [num_signals_per_class * seq_length, num_classes]

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.labels[idx], self.signals[idx]


def train_signal_generator(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    logging_frequency: int,
    verbose: bool,
    wandb_active: bool,
) -> Tuple[List[float], List[float], float]:

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    training_time = 0

    if wandb_active:
        wandb.watch(model, log="all")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for labels, signals in train_loader:
            optimizer.zero_grad()

            output, _ = model(labels)

            loss = criterion(
                output, signals
            )  # signals shape: [batch_size, seq_length, 1]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for labels, signals in val_loader:
                output, _ = model(labels)
                val_loss += criterion(output, signals).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % logging_frequency == 0:
            conditional_log(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {avg_val_loss:.4f}",
                verbose,
            )

        if wandb_active:
            wandb.log({"Train Loss": avg_train_loss, "Validation Loss": avg_val_loss})

    training_time = time.time() - start_time
    return train_losses, val_losses, training_time


def run_signal_benchmark(config_path: str):
    config = load_config(config_path)

    # Parameters from config
    signals_per_class = config["signals_per_class"]
    num_classes = config["num_classes"]
    seq_length = config["seq_length"]
    noise_std = config["noise_std"]
    output_directory = config["output_directory"]
    batch_size = config["batch_size"]
    hidden_size = config["hidden_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    logging_frequency = config["logging_frequency"]
    verbose_active = config.get("verbose", False)
    wandb_active = config.get("wandb", False)
    grad_clip = config.get("grad_clip", False)
    if grad_clip:
        grad_clip_norm = float(config["grad_clip_norm"])

    if wandb_active:
        wandb.init(project=config["wandb_project"], name=config["wandb_run"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    train_dataset = SignalGenerationDataset(
        signals_per_class, seq_length, num_classes, noise_std, device
    )
    validation_indices = list(range(0, len(train_dataset), 4))
    val_dataset = Subset(
        train_dataset, validation_indices
    )  # Since we want to overfit, we use the same data points from training

    # Plotting the dataset
    dataset_directory = os.path.join(output_directory, "dataset")
    os.makedirs(dataset_directory, exist_ok=True)
    for class_idx in range(num_classes):
        plt.figure(figsize=(20, 7))
        for i in range(
            class_idx * signals_per_class, (class_idx + 1) * signals_per_class
        ):
            plt.plot(train_dataset[i][1].to("cpu"))
        plt.title(f"Class {class_idx}")
        plt.savefig(os.path.join(dataset_directory, f"Signal_{class_idx}.png"))

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Models
    models = {
        "RNN": SignalRNN(num_classes, hidden_size, 1, seq_length, device),
        "LSTM": SignalRNN(num_classes, hidden_size, 1, seq_length, device),
        "GRU": SignalRNN(num_classes, hidden_size, 1, seq_length, device),
    }

    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")

        if grad_clip:
            print(f"Gradient clipping is active with norm {grad_clip_norm}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        train_losses, val_losses, training_time = train_signal_generator(
            model,
            train_loader,
            val_loader,
            epochs,
            learning_rate,
            logging_frequency,
            verbose_active,
            wandb_active,
        )

        results[model_name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "time": training_time,
        }

        print(f"{model_name} Training Time: {training_time:.2f} seconds")
        print(f"{model_name} Final Validation Loss: {val_losses[-1]:.6f}")

        # Save individual loss plots
        plot_losses(train_losses, val_losses, model_name, output_directory)

    # Create comparison plot
    plot_comparison(
        [results[m]["train_losses"] for m in models.keys()],
        [results[m]["val_losses"] for m in models.keys()],
        list(models.keys()),
        output_directory,
    )

    if wandb_active:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Signal Generation benchmark with config"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    args = parser.parse_args()
    run_signal_benchmark(args.config)
