from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedFlipFlopCell(nn.Module):
    """Optimized implementation of Flip-Flop cell with fused operations."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Fuse J and K gate parameters into single matrices for better parallelization
        self.gate_weights = nn.Parameter(
            torch.empty(
                (2 * hidden_size, input_size + hidden_size), device=device, dtype=dtype
            )
        )
        self.gate_bias = nn.Parameter(
            torch.zeros(2 * hidden_size, device=device, dtype=dtype)
        )

        # Output projection with improved initialization
        self.output_projection = nn.Parameter(
            torch.empty((output_size, hidden_size), device=device, dtype=dtype)
        )
        self.output_bias = nn.Parameter(
            torch.zeros(output_size, device=device, dtype=dtype)
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize parameters using Xavier uniform for better gradient flow."""
        nn.init.xavier_uniform_(self.gate_weights)
        nn.init.xavier_uniform_(self.output_projection)
        nn.init.constant_(self.gate_bias, 0.1)
        nn.init.zeros_(self.output_bias)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized forward pass with fused operations.

        Args:
            x: Input tensor (batch_size, input_size)
            h: Hidden state tensor (batch_size, hidden_size)

        Returns:
            (output, new_hidden_state) tuple
        """
        batch_size = x.size(0)
        if h is None:
            h = torch.zeros(
                batch_size, self.hidden_size, device=x.device, dtype=x.dtype
            )

        # Fused computation for J and K gates
        xh_concat = torch.cat([x, h], dim=1)
        gate_activations = F.linear(xh_concat, self.gate_weights, self.gate_bias)
        j, k = torch.chunk(torch.sigmoid(gate_activations), 2, dim=1)

        # Optimized hidden state update
        h_complement = 1 - h
        k_complement = 1 - k

        # FlipFlop update: j*(1-h) + (1-k)*h
        new_h = j * h_complement + k_complement * h

        # Compute output
        output = F.linear(new_h, self.output_projection, self.output_bias)

        return output, new_h


class OptimizedFlipFlopLayer(nn.Module):
    """Optimized RNN layer using FlipFlop cells with batch processing."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.cell = OptimizedFlipFlopCell(
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
        batch_size, seq_len, _ = x.size()

        # Pre-allocate output tensor
        outputs = torch.empty(
            batch_size, seq_len, self.cell.output_size, device=x.device, dtype=x.dtype
        )

        current_hidden = (
            hidden
            if hidden is not None
            else torch.zeros(
                batch_size, self.cell.hidden_size, device=x.device, dtype=x.dtype
            )
        )

        # Process sequence
        for t in range(seq_len):
            outputs[:, t, :], current_hidden = self.cell(x[:, t, :], current_hidden)

        return outputs, current_hidden
