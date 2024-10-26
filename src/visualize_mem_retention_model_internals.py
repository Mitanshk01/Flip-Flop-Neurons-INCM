import torch
import os
import argparse

from models.recurrent import SignalLSTM
from models.flipflop import SignalSligtlyOptimizedFlipFlopLayer
from common.utils import load_config
from train_memory_retention import MemoryRetentionDataset
from visualize_working_mem_model_internals import visualize_model_evolution


def main(config_path):
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Initialize models
    lstm_model = SignalLSTM(
        config["num_classes"],
        config["hidden_size"],
        1,
        config["seq_length"],
        device,
    )

    flipflop_model = SignalSligtlyOptimizedFlipFlopLayer(
        config["num_classes"],
        config["hidden_size"],
        1,
        config["seq_length"],
        device,
    )

    dataset = MemoryRetentionDataset(
        config["num_signals_for_actual_data"],
        config["num_signals_for_interference_data"],
        config["seq_length"],
        config["num_classes"],
        config["noise_std"],
        device,
    )

    # Visualize FlipFlop evolution
    flipflop_output_dir = os.path.join(
        config["output_directory"], "model_internals", "Flip Flop"
    )
    visualize_model_evolution(
        os.path.join(config["checkpoint_dir"], "Flip Flop"),
        flipflop_model,
        dataset,
        flipflop_output_dir,
        "FlipFlop",
        config["model_internals_viz_freq"],
        config["model_internals_viz_min_epoch_for_applying_freq"],
    )

    # Visualize LSTM evolution
    lstm_output_dir = os.path.join(
        config["output_directory"], "model_internals", "LSTM"
    )
    visualize_model_evolution(
        os.path.join(config["checkpoint_dir"], "LSTM"),
        lstm_model,
        dataset,
        lstm_output_dir,
        "LSTM",
        config["model_internals_viz_freq"],
        config["model_internals_viz_min_epoch_for_applying_freq"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Signal Generation benchmark with config"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    args = parser.parse_args()
    main(args.config)
