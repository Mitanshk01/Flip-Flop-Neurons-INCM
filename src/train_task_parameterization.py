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
from models.optim_flipflop import SignalOptimizedFlipFlopLayer
from common.utils import seed_models, load_config, get_num_params, save_dict_as_json
from common.plotting import plot_losses, plot_comparison
from common.logging import conditional_log
from train_working_memory import WorkingMemoryDataset, train_working_memory_capacity

seed_models()


def run_working_task_param_benchmark(config_path: str):
    config = load_config(config_path)

    # Parameters from config
    num_classes = config["num_classes"]
    seq_length = config["seq_length"]
    noise_std = config["noise_std"]
    output_directory = config["output_directory"]
    batch_size = config["batch_size"]
    hidden_size = config["hidden_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    logging_frequency = config["logging_frequency"]
    num_signals_per_class = config["num_signals_per_class"]
    verbose_active = config.get("verbose", False)
    wandb_active = config.get("wandb", False)
    grad_clip = config.get("grad_clip", False)

    if grad_clip:
        grad_clip_norm = float(config["grad_clip_norm"])

    if wandb_active:
        wandb.init(project=config["wandb_project"], name=config["wandb_run"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    train_dataset = WorkingMemoryDataset(
        num_signals_per_class,
        seq_length,
        num_classes,
        noise_std,
        device,
    )
    validation_indices = list(range(0, len(train_dataset), 4))
    val_dataset = Subset(
        train_dataset, validation_indices
    )  # Since we want to overfit, we use the same data points from training

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Models
    models = {
        "RNN": SignalRNN(num_classes, hidden_size, 1, seq_length, device),
        "LSTM": SignalLSTM(num_classes, hidden_size, 1, seq_length, device),
        "GRU": SignalGRU(num_classes, hidden_size, 1, seq_length, device),
        "Flip Flop": SignalOptimizedFlipFlopLayer(
            num_classes, hidden_size, 1, seq_length, device
        ),
    }

    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")

        if grad_clip:
            print(f"Gradient clipping is active with norm {grad_clip_norm}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        train_losses, val_losses, training_time = train_working_memory_capacity(
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
            "time": float(training_time),
            "num_params": int(num_params),
        }

        print(f"{model_name} Training Time: {training_time:.2f} seconds")
        print(f"{model_name} Number of Parameters: {num_params}")
        print(f"{model_name} Best Training Loss: {np.min(val_losses):.6f}")
        print(f"{model_name} Best Validation Loss: {np.min(train_losses):.6f}")

        # Save individual loss plots
        plot_losses(train_losses, val_losses, model_name, output_directory)

    # Create comparison plot
    plot_comparison(
        [results[m]["train_losses"] for m in models.keys()],
        [results[m]["val_losses"] for m in models.keys()],
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
        description="Run Signal Generation benchmark with config"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    args = parser.parse_args()
    run_working_task_param_benchmark(args.config)
