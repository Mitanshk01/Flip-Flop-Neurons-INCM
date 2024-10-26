import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import time
import wandb
import argparse
import os
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from models.recurrent import SentimentRNN, SentimentLSTM, SentimentGRU
from models.optim_flipflop import SentimentOptimizedFlipFlopLayer
from common.utils import seed_models, load_config, get_num_params, save_dict_as_json
from common.plotting import (
    plot_losses,
    plot_comparison,
    plot_accuracies,
    plot_accuracy_comparison,
)
from common.logging import conditional_log

seed_models()


class IMDBDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_length: int,
        device: torch.device,
    ):
        X_padded = sequence.pad_sequences(X, maxlen=max_length)

        self.sequences = torch.tensor(X_padded, dtype=torch.long).to(device)
        self.labels = torch.tensor(y, dtype=torch.float32).to(device)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def train_sentiment_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    logging_frequency: int,
    verbose: bool,
    wandb_active: bool,
) -> Tuple[List[float], List[float], List[float], List[float], float]:

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    training_time = 0

    if wandb_active:
        wandb.watch(model, log="all")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for sequences, labels in train_loader:
            optimizer.zero_grad()
            output, _ = model(sequences)
            loss = criterion(output.squeeze(), labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = (output.squeeze() > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                output, _ = model(sequences)
                val_loss += criterion(output.squeeze(), labels).item()
                predicted = (output.squeeze() > 0.5).long()
                total += labels.size(0)
                correct += (predicted == labels.long()).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        if (epoch + 1) % logging_frequency == 0 and verbose:
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.2%}, "
                f"Validation Loss: {avg_val_loss:.4f}, "
                f"Validation Accuracy: {val_accuracy:.2%}"
            )

        if wandb_active:
            wandb.log(
                {
                    "Train Loss": avg_train_loss,
                    "Train Accuracy": train_accuracy,
                    "Validation Loss": avg_val_loss,
                    "Validation Accuracy": val_accuracy,
                }
            )

    training_time = time.time() - start_time
    return train_losses, val_losses, train_accuracies, val_accuracies, training_time


def run_sentiment_benchmark(config_path: str):
    config = load_config(config_path)

    # Parameters from config
    vocab_size = config["vocab_size"]
    max_length = config["max_length"]
    embedding_dim = config["embedding_dim"]
    output_directory = config["output_directory"]
    hidden_size = config["hidden_size"]
    batch_size = config["batch_size"]
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

    # Load IMDB dataset
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

    # Create datasets
    train_dataset = IMDBDataset(X_train, y_train, max_length, device)
    val_dataset = IMDBDataset(X_test, y_test, max_length, device)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Models
    models = {
        "RNN": SentimentRNN(vocab_size, embedding_dim, hidden_size, device),
        "LSTM": SentimentLSTM(vocab_size, embedding_dim, hidden_size, device),
        "GRU": SentimentGRU(vocab_size, embedding_dim, hidden_size, device),
        "Flip Flop": SentimentOptimizedFlipFlopLayer(
            vocab_size, embedding_dim, hidden_size, device
        ),
    }

    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")

        (
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            training_time,
        ) = train_sentiment_model(
            model,
            train_loader,
            val_loader,
            epochs,
            learning_rate,
            logging_frequency,
            verbose_active,
            wandb_active,
        )

        num_params = get_num_params(model)

        results[model_name] = {
            "train_losses": [float(val) for val in train_losses],
            "val_losses": [float(val) for val in val_losses],
            "train_accuracies": [float(val) for val in train_accuracies],
            "val_accuracies": [float(val) for val in val_accuracies],
            "min_train_loss": {
                "epoch": int(
                    np.max(
                        [
                            i
                            for i in range(len(train_losses))
                            if train_losses[i] == np.min(train_losses)
                        ]
                    )
                ),
                "value": float(np.min(train_losses)),
            },
            "min_val_loss": {
                "epoch": int(
                    np.max(
                        [
                            i
                            for i in range(len(val_losses))
                            if val_losses[i] == np.min(val_losses)
                        ]
                    )
                ),
                "value": float(np.min(val_losses)),
            },
            "max_train_accuracy": {
                "epoch": int(
                    np.max(
                        [
                            i
                            for i in range(len(train_accuracies))
                            if train_accuracies[i] == np.max(train_accuracies)
                        ]
                    )
                ),
                "value": float(np.max(train_accuracies)),
            },
            "max_val_accuracy": {
                "epoch": int(
                    np.max(
                        [
                            i
                            for i in range(len(val_accuracies))
                            if val_accuracies[i] == np.max(val_accuracies)
                        ]
                    )
                ),
                "value": float(np.max(val_accuracies)),
            },
            "time": float(training_time),
            "num_params": int(num_params),
        }

        print(f"{model_name} Training Time: {training_time:.2f} seconds")
        print(f"{model_name} Number of Parameters: {num_params}")
        print(f"{model_name} Best Training Loss: {np.min(val_losses):.6f}")
        print(f"{model_name} Best Validation Loss: {np.min(train_losses):.6f}")
        print(f"{model_name} Best Training Accuracy: {np.max(train_accuracies):.6f}")
        print(f"{model_name} Best Validation Accuracy: {np.max(val_accuracies):.6f}")

        plot_losses(train_losses, val_losses, model_name, output_directory)
        plot_accuracies(train_accuracies, val_accuracies, model_name, output_directory)

    plot_comparison(
        [results[m]["train_losses"] for m in models.keys()],
        [results[m]["val_losses"] for m in models.keys()],
        list(models.keys()),
        output_directory,
    )

    plot_accuracy_comparison(
        [results[m]["train_accuracies"] for m in models.keys()],
        [results[m]["val_accuracies"] for m in models.keys()],
        list(models.keys()),
        output_directory,
    )

    # Save results
    save_dict_as_json(results, os.path.join(output_directory, "results.json"))
    print("Saved results.")

    if wandb_active:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run IMDB Sentiment Analysis benchmark with config"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    args = parser.parse_args()
    run_sentiment_benchmark(args.config)
