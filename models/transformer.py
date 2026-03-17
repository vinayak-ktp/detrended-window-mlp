import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                    # (max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()   # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, n_features, forecast_len, d_model, nhead, num_layers, dim_ff=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,                # Attention heads
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.fc = nn.Linear(d_model, forecast_len)

    def forward(self, x):
        x = self.input_proj(x)      # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)     # (batch_size, seq_len, d_model)
        out = self.encoder(x)       # (batch_size, seq_len, d_model)
        summary = out[:, -1, :]     # (batch_size, d_model)
        return self.fc(summary)     # (batch_size, forecast_len)
