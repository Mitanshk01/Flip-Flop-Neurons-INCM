from typing import Tuple, Optional
import torch
import torch.nn as nn


class FlipFlopCell(nn.Module):
    """Implementation of Flip-Flop cell in PyTorch."""

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
        self.device = device
        self.dtype = dtype

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

    def _compute_gate(
        self, x: torch.Tensor, h: torch.Tensor, x_layer: nn.Linear, h_layer: nn.Linear
    ) -> torch.Tensor:
        """Compute gate activation."""
        return torch.sigmoid(x_layer(x) + h_layer(h))

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the FlipFlop cell.

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

        j = self._compute_gate(x, h, self.j_linear_x, self.j_linear_h)
        k = self._compute_gate(x, h, self.k_linear_x, self.k_linear_h)

        # Update hidden state using FlipFlop logic
        new_h = j * (1 - h) + (1 - k) * h

        # Project hidden state to output space
        output = self.output_projection(new_h)

        return output, new_h


class FlipFlopLayer(nn.Module):
    """RNN layer using FlipFlop cells."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.cell = FlipFlopCell(input_size, hidden_size, output_size, device, dtype)
        self.device = device
        self.dtype = dtype

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the FlipFlop layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            hidden (Optional[torch.Tensor]): Initial hidden state

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (outputs, final_hidden_state)
                outputs: tensor of shape (batch_size, seq_len, output_size)
                final_hidden_state: tensor of shape (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            hidden = torch.zeros(
                batch_size, self.cell.hidden_size, device=self.device, dtype=self.dtype
            )

        outputs = []
        current_hidden = hidden

        for t in range(seq_len):
            output, current_hidden = self.cell(x[:, t, :], current_hidden)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        return outputs, current_hidden


class SignalSligtlyOptimizedFlipFlopLayer(nn.Module):
    """RNN layer using FlipFlop cells with batch processing."""

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
        self.cell = FlipFlopCell(input_size, hidden_size, output_size, device, dtype)

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
