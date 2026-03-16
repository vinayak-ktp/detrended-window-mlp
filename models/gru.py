import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, n_features, hidden_dim, forecast_len, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, forecast_len)

    def forward(self, x):
        out, _ = self.gru(x)    # out: (batch_size, seq_len, hidden_size)
        last = out[: -1 :]      # last hidden state: (batch_size, hidden_size)
        return self.fc(last)    # output: (batch_size, forecast_len)
