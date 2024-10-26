from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlipFlopCellSTDP(nn.Module):
    """Implementation of Flip-Flop cell with Spike-Timing-Dependent Plasticity in PyTorch."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        stdp_window: float = 20.0,  # Time window for STDP in milliseconds
        learning_rate: float = 0.01,  # Learning rate for weight updates
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.dtype = dtype
        self.stdp_window = stdp_window
        self.learning_rate = learning_rate

        # Gates with separate layers
        self.j_linear_x = nn.Linear(
            self.input_size, self.hidden_size, device=self.device, dtype=self.dtype
        )
        self.j_linear_h = nn.Linear(
            self.hidden_size, self.hidden_size, device=self.device, dtype=self.dtype
        )
        self.k_linear_x = nn.Linear(
            self.input_size, self.hidden_size, device=self.device, dtype=self.dtype
        )
        self.k_linear_h = nn.Linear(
            self.hidden_size, self.hidden_size, device=self.device, dtype=self.dtype
        )

        # Output projection
        self.output_projection = nn.Linear(
            self.hidden_size, self.output_size, device=self.device, dtype=self.dtype
        )

        # Store spike timings for STDP updates
        self.pre_spike_times = torch.zeros(
            1, self.hidden_size, device=self.device, dtype=self.dtype
        )
        self.post_spike_times = torch.zeros(
            1, self.hidden_size, device=self.device, dtype=self.dtype
        )

    def _compute_gate(
        self, x: torch.Tensor, h: torch.Tensor, x_layer: nn.Linear, h_layer: nn.Linear
    ) -> torch.Tensor:
        """Compute gate activation."""
        return torch.sigmoid(x_layer(x) + h_layer(h))

    def apply_stdp(self, pre_spike: torch.Tensor, post_spike: torch.Tensor) -> None:
        """Apply STDP rule to adjust weights based on spike timing."""
        # Calculate the time difference between pre and post-synaptic spikes
        time_diff = post_spike - pre_spike

        # If pre-synaptic spike is earlier, apply LTP (potentiation)
        potentiation = time_diff < self.stdp_window
        # If post-synaptic spike is earlier, apply LTD (depression)
        depression = time_diff > -self.stdp_window

        # Update the weights based on potentiation and depression
        if potentiation.any():
            self.j_linear_x.weight.data += (
                self.learning_rate * pre_spike[potentiation].mean()
            )
            self.j_linear_h.weight.data += (
                self.learning_rate * pre_spike[potentiation].mean()
            )

        if depression.any():
            self.k_linear_x.weight.data -= (
                self.learning_rate * post_spike[depression].mean()
            )
            self.k_linear_h.weight.data -= (
                self.learning_rate * post_spike[depression].mean()
            )

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the FlipFlop cell with STDP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            h (Optional[torch.Tensor]): Hidden state tensor of shape (batch_size, hidden_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (output, new_hidden_state)
                output: tensor of shape (batch_size, output_size)
                new_hidden_state: tensor of shape (batch_size, hidden_size)
        """
        if h is None:
            h = torch.zeros(
                x.size(0), self.hidden_size, device=self.device, dtype=self.dtype
            )

        # Record pre- and post-synaptic spike timings
        pre_spike_time = torch.rand(x.size(0), self.hidden_size, device=self.device)
        post_spike_time = torch.rand(x.size(0), self.hidden_size, device=self.device)

        # Compute gates and hidden states
        j = self._compute_gate(x, h, self.j_linear_x, self.j_linear_h)
        k = self._compute_gate(x, h, self.k_linear_x, self.k_linear_h)

        # Update hidden state using FlipFlop logic
        new_h = j * (1 - h) + (1 - k) * h

        # Apply STDP rule based on spike timing
        self.apply_stdp(pre_spike_time, post_spike_time)

        # Project hidden state to output space
        output = self.output_projection(new_h)

        return output, new_h


class SignalFlipFlopLayerSTDP(nn.Module):
    """Optimized RNN layer using FlipFlop cells with batch processing."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seq_length: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.cell = FlipFlopCellSTDP(
            input_size, hidden_size, output_size, device, dtype
        )

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized forward pass

        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            hidden: Initial hidden state

        Returns:
            (outputs, final_hidden_state) tuple
        """
        batch_size = x.size(0)

        x = x.unsqueeze(1).repeat(
            1, self.seq_length, 1
        )  # [batch_size, seq_length, num_classes]

        # Pre-allocate output tensor
        outputs = torch.empty(
            batch_size,
            self.seq_length,
            self.cell.output_size,
            device=x.device,
            dtype=x.dtype,
        )

        current_hidden = (
            hidden
            if hidden is not None
            else torch.zeros(
                batch_size, self.cell.hidden_size, device=x.device, dtype=x.dtype
            )
        )

        # Process sequence
        for t in range(self.seq_length):
            outputs[:, t, :], current_hidden = self.cell(x[:, t, :], current_hidden)

        return outputs, current_hidden
