import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import time
import wandb
import argparse
from models.flipflop import FlipFlopLayer
from models.recurrent import VanillaRNN
from models.optim_flipflop import OptimizedFlipFlopLayer
from common.utils import seed_models, load_config
from common.plotting import plot_losses, plot_comparison
from common.logging import conditional_log

seed_models()


class SequenceDataset(Dataset):
    def __init__(
        self, num_sequences: int, seq_length: int, input_size: int, device: torch.device
    ):
        self.sequences = []
        self.targets = []

        for _ in range(num_sequences):
            freqs = torch.rand(input_size) * 2 * np.pi

            t = torch.linspace(0, 4 * np.pi, seq_length).unsqueeze(-1)

            sequence = torch.sin(t * freqs)

            target = torch.sin((t + 0.1) * freqs)

            self.sequences.append(sequence)
            self.targets.append(target)

        self.sequences = torch.stack(self.sequences).to(device)
        self.targets = torch.stack(self.targets).to(device)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


# Training function
def train_model(
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
        for batch_x, batch_y in train_loader:
            print(batch_x.shape, batch_y.shape)
            optimizer.zero_grad()
            output, _ = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output, _ = model(batch_x)
                val_loss += criterion(output, batch_y).item()
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


def run_benchmark(config_path: str):
    config = load_config(config_path)

    # Parameters from config
    input_size = config["input_size"]
    hidden_size = config["hidden_size"]
    output_size = config["output_size"]
    seq_length = config["seq_length"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    logging_frequency = config["logging_frequency"]
    output_directory = config["output_directory"]
    training_size = config["training_size"]
    validation_size = config["validation_size"]
    verbose_active = config.get("verbose", False)
    wandb_active = config.get("wandb", False)
    if wandb_active:
        wandb.init(project=config["wandb_project"], name=config["wandb_run"])

    # Assign device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Create models
    flipflop_model = FlipFlopLayer(input_size, hidden_size, output_size, device)
    optim_flipflop_model = OptimizedFlipFlopLayer(
        input_size, hidden_size, output_size, device
    )
    rnn_model = VanillaRNN(input_size, hidden_size, output_size, device)

    # Create datasets
    train_dataset = SequenceDataset(training_size, seq_length, input_size, device)
    val_dataset = SequenceDataset(validation_size, seq_length, input_size, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Train models
    print("Training FlipFlop model...")
    ff_train_losses, ff_val_losses, ff_time = train_model(
        flipflop_model,
        train_loader,
        val_loader,
        epochs,
        learning_rate,
        logging_frequency,
        verbose_active,
        wandb_active,
    )

    print("Training Optimized FlipFlop model...")
    optim_ff_train_losses, optim_ff_val_losses, optim_ff_time = train_model(
        optim_flipflop_model,
        train_loader,
        val_loader,
        epochs,
        learning_rate,
        logging_frequency,
        verbose_active,
        wandb_active,
    )

    print("Training RNN model...")
    rnn_train_losses, rnn_val_losses, rnn_time = train_model(
        rnn_model,
        train_loader,
        val_loader,
        epochs,
        learning_rate,
        logging_frequency,
        verbose_active,
        wandb_active,
    )

    print(f"\nTraining Times:")
    print(f"FlipFlop: {ff_time:.2f} seconds")
    print(f"Optimized FlipFlop: {optim_ff_time:.2f} seconds")
    print(f"RNN: {rnn_time:.2f} seconds")

    print(f"\nFinal Validation Losses:")
    print(f"FlipFlop: {ff_val_losses[-1]:.6f}")
    print(f"Optimized FlipFlop: {optim_ff_val_losses[-1]:.6f}")
    print(f"RNN: {rnn_val_losses[-1]:.6f}")

    # Save loss plots
    plot_losses(ff_train_losses, ff_val_losses, "FlipFlop", output_directory)
    plot_losses(
        optim_ff_train_losses,
        optim_ff_val_losses,
        "Optimized_FlipFlop",
        output_directory,
    )
    plot_losses(rnn_train_losses, rnn_val_losses, "RNN", output_directory)
    plot_comparison(
        [ff_train_losses, optim_ff_train_losses, rnn_train_losses],
        [ff_val_losses, optim_ff_val_losses, rnn_val_losses],
        ["FlipFlop", "Optimized_FlipFlop", "RNN"],
        output_directory,
    )

    if wandb_active:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RNN benchmark with config")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    args = parser.parse_args()
    run_benchmark(args.config)
