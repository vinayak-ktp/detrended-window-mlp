import torch.nn as nn
import torch.nn.functional as F


class CasualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.conv_layers = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
        )
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        conv_out = self.conv_layers(x)
        projection = self.residual_proj(x)
        return F.relu(conv_out + projection)


class TCNModel(nn.Module):
    def __init__(self, n_features, num_channels, kernel_size, forecast_len, dropout=0.0):
        super().__init__()
        layers = []
        in_channels = n_features
        for i, out_channels in enumerate(num_channels):
            dilation = 2**i
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
            in_channels = out_channels
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Linear(num_channels[-1], forecast_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)             # (batch, num_channels[-1], seq_len)
        x = x[:, :, -1]             # (batch, num_channels[-1])
        return self.fc(x)           # (batch, forecast_len)
