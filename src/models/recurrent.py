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


class SignalRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seq_length: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, device=self.device)
        self.output_layer = nn.Linear(hidden_size, output_size, device=self.device)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        x = x.unsqueeze(1).repeat(
            1, self.seq_length, 1
        )  # [batch_size, seq_length, num_classes]

        if h is None:
            h = torch.zeros(1, batch_size, self.hidden_size, device=self.device)

        output, hidden = self.rnn(x, h)  # output: [batch_size, seq_length, hidden_size]
        output = self.output_layer(
            output
        )  # output: [batch_size, seq_length, output_size]
        return output, hidden


class SignalLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seq_length: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(
            input_size, hidden_size, batch_first=True, device=self.device
        )
        self.output_layer = nn.Linear(hidden_size, output_size, device=self.device)

    def forward(
        self, x: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.size(0)

        x = x.unsqueeze(1).repeat(
            1, self.seq_length, 1
        )  # [batch_size, seq_length, input_size]

        if h is None:
            h = (
                torch.zeros(1, batch_size, self.hidden_size, device=self.device),
                torch.zeros(1, batch_size, self.hidden_size, device=self.device),
            )

        output, hidden = self.lstm(
            x, h
        )  # output: [batch_size, seq_length, hidden_size]
        output = self.output_layer(
            output
        )  # output: [batch_size, seq_length, output_size]
        return output, hidden


class SignalGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seq_length: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, device=self.device)
        self.output_layer = nn.Linear(hidden_size, output_size, device=self.device)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        x = x.unsqueeze(1).repeat(
            1, self.seq_length, 1
        )  # [batch_size, seq_length, input_size]

        if h is None:
            h = torch.zeros(1, batch_size, self.hidden_size, device=self.device)

        output, hidden = self.gru(x, h)  # output: [batch_size, seq_length, hidden_size]
        output = self.output_layer(
            output
        )  # output: [batch_size, seq_length, output_size]
        return output, hidden


class SentimentRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.to(device)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            h = torch.zeros(1, x.size(0), self.hidden_size, device=self.device)

        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, h)
        output = self.fc(output[:, -1, :])
        return self.sigmoid(output), hidden


class SentimentLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.to(device)

    def forward(
        self, x: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if h is None:
            h = (
                torch.zeros(1, x.size(0), self.hidden_size, device=self.device),
                torch.zeros(1, x.size(0), self.hidden_size, device=self.device),
            )

        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, h)
        output = self.fc(output[:, -1, :])
        return self.sigmoid(output), hidden


class SentimentGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.to(device)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            h = torch.zeros(1, x.size(0), self.hidden_size, device=self.device)

        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, h)
        output = self.fc(output[:, -1, :])
        return self.sigmoid(output), hidden
