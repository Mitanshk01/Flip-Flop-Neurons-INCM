import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from typing import Dict, List, Tuple
import seaborn as sns
from common.utils import load_config
import argparse
import glob
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from models.recurrent import SignalLSTM
from models.flipflop import SignalSligtlyOptimizedFlipFlopLayer
from train_working_memory import WorkingMemoryDataset


def load_checkpoint(path: str, model: torch.nn.Module) -> Dict:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def get_internal_states(
    model: torch.nn.Module, input_data: torch.Tensor, signals: torch.Tensor
) -> Dict:
    model.eval()
    states = {}

    with torch.no_grad():
        if isinstance(model, SignalLSTM):
            output, (h_n, c_n) = model(input_data)
            states = {
                "hidden_states": h_n.squeeze(0).cpu().numpy(),
                "cell_states": c_n.squeeze(0).cpu().numpy(),
                "outputs": output.cpu().numpy(),
                "target_signals": signals.cpu().numpy(),
                "hidden_grad_norm": torch.norm(
                    h_n.grad if h_n.grad is not None else torch.zeros_like(h_n)
                ).item(),
                "cell_grad_norm": torch.norm(
                    c_n.grad if c_n.grad is not None else torch.zeros_like(c_n)
                ).item(),
            }

            states["hidden_stats"] = {
                "mean": np.mean(states["hidden_states"], axis=0),
                "std": np.std(states["hidden_states"], axis=0),
                "sparsity": np.mean(states["hidden_states"] == 0),
            }
        elif isinstance(model, SignalSligtlyOptimizedFlipFlopLayer):
            output, hidden = model(input_data)
            states = {
                "hidden_states": hidden.cpu().numpy(),
                "outputs": output.cpu().numpy(),
                "target_signals": signals.cpu().numpy(),
                "j_gates": [],
                "k_gates": [],
            }

            for t in range(model.seq_length):
                j = (
                    model.cell._compute_gate(
                        input_data,
                        hidden,
                        model.cell.j_linear_x,
                        model.cell.j_linear_h,
                    )
                    .cpu()
                    .numpy()
                )
                k = (
                    model.cell._compute_gate(
                        input_data,
                        hidden,
                        model.cell.k_linear_x,
                        model.cell.k_linear_h,
                    )
                    .cpu()
                    .numpy()
                )
                states["j_gates"].append(j)
                states["k_gates"].append(k)

            states["j_gates"] = np.stack(states["j_gates"])
            states["k_gates"] = np.stack(states["k_gates"])

            # states["gate_stats"] = {
            #     "j_mean": np.mean(states["j_gates"]),
            #     "k_mean": np.mean(states["k_gates"]),
            #     "j_std": np.std(states["j_gates"]),
            #     "k_std": np.std(states["k_gates"]),
            #     "gate_correlation": pearsonr(
            #         states["j_gates"].flatten(), states["k_gates"].flatten()
            #     )[0],
            # }

    hidden_2d = PCA(n_components=2).fit_transform(states["hidden_states"])
    states["hidden_pca"] = hidden_2d

    states["error_analysis"] = {
        "mse": np.mean((states["outputs"] - states["target_signals"]) ** 2),
        "mae": np.mean(np.abs(states["outputs"] - states["target_signals"])),
    }
    return states


def plot_basic_state_evolution(
    states: Dict, epoch: int, model_type: str, save_path: str
):
    plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2)
    plt.subplot(gs[0, 0])
    sns.heatmap(states["hidden_states"], cmap="viridis")
    plt.title(f"Hidden States at Epoch {epoch}")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Batch Sample")

    # Plot output trajectories
    plt.subplot(gs[0, 1])
    for i in range(min(20, states["outputs"].shape[0])):
        plt.plot(states["outputs"][i, :, 0], label=f"Sample {i}")
    plt.title(f"Output Trajectories at Epoch {epoch}")
    plt.xlabel("Time Step")
    plt.ylabel("Output Value")
    plt.legend()

    if model_type == "LSTM":
        # Plot cell states
        plt.subplot(gs[1, :])
        sns.heatmap(states["cell_states"], cmap="viridis")
        plt.title(f"Cell States at Epoch {epoch}")
        plt.xlabel("Hidden Dimension")
        plt.ylabel("Batch Sample")

    elif model_type == "FlipFlop":
        # Plot J-K gate activations
        plt.subplot(gs[1, 0])
        sns.heatmap(states["j_gates"].mean(axis=1), cmap="viridis")
        plt.title(f"Average J Gate Activations at Epoch {epoch}")
        plt.xlabel("Hidden Dimension")
        plt.ylabel("Time Step")

        plt.subplot(gs[1, 1])
        sns.heatmap(states["k_gates"].mean(axis=1), cmap="viridis")
        plt.title(f"Average K Gate Activations at Epoch {epoch}")
        plt.xlabel("Hidden Dimension")
        plt.ylabel("Time Step")

    # Plot state trajectory for selected neurons
    plt.subplot(gs[2, :])
    selected_neurons = [0, 16, 32, 48]
    for neuron in selected_neurons:
        plt.plot(states["hidden_states"][:, neuron], label=f"Neuron {neuron}")
    plt.title(f"Hidden State Trajectories for Selected Neurons at Epoch {epoch}")
    plt.xlabel("Batch Sample")
    plt.ylabel("Activation")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"internal_states_basic", f"epoch_{epoch}.png"))
    plt.close()


def plot_advanced_state_evolution(
    states: Dict, epoch: int, model_type: str, save_path: str
):
    plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2)

    # Plot PCA visualization
    plt.subplot(gs[0, 0])
    plt.scatter(states["hidden_pca"][:, 0], states["hidden_pca"][:, 1])
    plt.title(f"PCA of Hidden States at Epoch {epoch}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # Plot error distribution
    plt.subplot(gs[0, 1])
    errors = (states["outputs"] - states["target_signals"]).flatten()
    plt.hist(errors, bins=50)
    plt.title(f"Error Distribution at Epoch {epoch}")
    plt.xlabel("Error")
    plt.ylabel("Count")

    if model_type == "LSTM":
        # Plot hidden-cell state correlation
        plt.subplot(gs[1, 0])
        mask = ~np.isnan(states["hidden_states"]) & ~np.isnan(states["cell_states"])
        hidden_states_clean = states["hidden_states"][mask]
        cell_states_clean = states["cell_states"][mask]
        correlation_matrix = np.corrcoef(hidden_states_clean.T, cell_states_clean.T)
        sns.heatmap(correlation_matrix, cmap="coolwarm")
        plt.title("Hidden-Cell State Correlations")

        # Plot activation statistics
        plt.subplot(gs[1, 1])
        plt.bar(
            ["Mean", "Std", "Sparsity"],
            [
                states["hidden_stats"]["mean"].mean(),
                states["hidden_stats"]["std"].mean(),
                states["hidden_stats"]["sparsity"],
            ],
        )
        plt.title("Hidden State Statistics")

    elif model_type == "FlipFlop":
        # Plot gate activation distributions
        plt.subplot(gs[1, 0])
        plt.hist(states["j_gates"].flatten(), alpha=0.5, label="J gates")
        plt.hist(states["k_gates"].flatten(), alpha=0.5, label="K gates")
        plt.title("Gate Activation Distributions")
        plt.legend()

        # Plot gate correlation over time
        plt.subplot(gs[1, 1])
        gate_corr = [
            pearsonr(j.flatten(), k.flatten())[0]
            for j, k in zip(states["j_gates"], states["k_gates"])
        ]
        plt.plot(gate_corr)
        plt.title("Gate Correlation over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Correlation")

        # Plot gate correlation matrix
        plt.subplot(gs[2, 0])
        last_j, last_k = states["j_gates"][-1], states["k_gates"][-1]
        mask = ~np.isnan(last_j) & ~np.isnan(last_k)
        last_j_clean = last_j[mask]
        last_k_clean = last_k[mask]
        correlation_matrix = np.corrcoef(last_j_clean.T, last_k_clean.T)
        sns.heatmap(correlation_matrix, cmap="coolwarm")
        plt.title("J and K States Correlations (last timestep)")

    # Plot error analysis
    plt.subplot(gs[2, 1])
    metrics = list(states["error_analysis"].keys())
    values = list(states["error_analysis"].values())
    plt.bar(metrics, values)
    plt.title(f"Error Metrics at Epoch {epoch}")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, "internal_states_advanced", f"epoch_{epoch}.png")
    )
    plt.close()


def plot_temporal_state_evolution(
    states: Dict, epoch: int, model_type: str, save_path: str
):
    plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2)

    # Plot temporal evolution of hidden states
    plt.subplot(gs[0, :])
    plt.imshow(states["hidden_states"].T, aspect="auto", cmap="viridis")
    plt.title(f"Temporal Evolution of Hidden States at Epoch {epoch}")
    plt.xlabel("Sample")
    plt.ylabel("Hidden Unit")
    plt.colorbar()

    # Plot cross-correlation between output and target
    plt.subplot(gs[1, :])
    crosscorr = np.correlate(
        states["outputs"].mean(axis=(0, 2)),
        states["target_signals"].mean(axis=(0, 2)),
        mode="full",
    )
    plt.plot(crosscorr)
    plt.title("Output-Target Cross-correlation")
    plt.xlabel("Lag")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"internal_states_temporal", f"epoch_{epoch}.png")
    )
    plt.close()


def plot_state_evolution(states: Dict, epoch: int, model_type: str, save_path: str):
    plot_basic_state_evolution(states, epoch, model_type, save_path)
    plot_advanced_state_evolution(states, epoch, model_type, save_path)
    plot_temporal_state_evolution(states, epoch, model_type, save_path)


def visualize_model_evolution(
    checkpoint_dir: str,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    output_dir: str,
    model_type: str,
    model_internals_viz_freq: int,
    model_internals_viz_min_epoch_for_applying_freq,
):
    """Visualize model internal states across training epochs."""
    os.makedirs(output_dir, exist_ok=True)

    # Making the other dirs
    os.makedirs(os.path.join(output_dir, "internal_states_basic"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "internal_states_advanced"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "internal_states_temporal"), exist_ok=True)

    # Get all checkpoints sorted by epoch
    unfiltered_checkpoint_files = sorted(
        glob.glob(os.path.join(checkpoint_dir, "*.pt")),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    checkpoint_files = [
        file
        for file in unfiltered_checkpoint_files
        if int(file.split("_")[-1].split(".")[0])
        < model_internals_viz_min_epoch_for_applying_freq
        or int(file.split("_")[-1].split(".")[0]) % int(model_internals_viz_freq) == 0
    ]

    # Get a fixed batch of data for visualization
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False)
    labels, signals = next(iter(dataloader))

    for checkpoint_file in checkpoint_files:
        epoch = int(checkpoint_file.split("_")[-1].split(".")[0])

        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_file, model)

        # Get internal states
        states = get_internal_states(model, labels, signals)

        # Create visualization
        plot_state_evolution(states, epoch, model_type, output_dir)

        print(f"Processed epoch {epoch}")


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

    dataset = WorkingMemoryDataset(
        config["num_signals_per_class"],
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
