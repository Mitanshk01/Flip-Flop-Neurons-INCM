import torch
from typing import Tuple, Optional
import torch.nn as nn


class VanillaRNN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, device: torch.device
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size, hidden_size, batch_first=True, device=self.device
        ).to(self.device)
        self.output_layer = nn.Linear(hidden_size, output_size, device=self.device)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            h = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)

        output, hidden = self.rnn(x, h)
        output = self.output_layer(output)
        return output, hidden
