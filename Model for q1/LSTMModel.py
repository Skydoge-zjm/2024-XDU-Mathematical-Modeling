import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight_f = nn.Linear(dim, 1)

    def forward(self, x):
        weight = self.weight_f(x)
        return x * weight


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=None, num_layers=1):
        super().__init__()
        output_size = output_size if output_size is not None else input_size
        self.linear1 = nn.Linear(input_size, hidden_size)

        self.relu1 = nn.LeakyReLU()
        self.attention = Attention(dim=hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x0):
        x1 = self.linear1(x0)
        x1 = self.relu1(x1)
        x1 = self.attention(x1)
        x1, _ = self.lstm(x1)
        x2 = self.linear2(x1)
        return x2[:, -1, :]


