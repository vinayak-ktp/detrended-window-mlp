import torch
import torch.nn as nn


def detrend_batch(x):
    batch, seq_len, n_features = x.shape

    t = torch.arange(seq_len, dtype=torch.float32, device=x.device)
    t_mean = t.mean()
    t_centered = t - t_mean
    t_3d = t_centered.reshape(1, seq_len, 1)

    y_mean = x.mean(dim=1)    # (batch, n_features)

    y_mean_3d = y_mean.reshape(batch, 1, n_features)
    x_centered = x - y_mean_3d    # (batch, seq_len, n_features)

    num = (t_3d * x_centered).sum(dim=1)     # (batch, n_features)
    den = (t_centered ** 2).sum()            # scalar

    slopes = num / den                       # (batch, n_features)
    intercepts = y_mean - slopes * t_mean    # (batch, n_features)

    slopes_3d = slopes.reshape(batch, 1, n_features)
    intercepts_3d = intercepts.reshape(batch, 1, n_features)

    trend = t_3d * slopes_3d + intercepts_3d    # (batch, seq_len, n_features)
    residuals = x - trend                       # (batch, seq_len, n_features)

    # extrapolate one step ahead
    t_next = seq_len - t_mean                    # scalar
    trend_at_T = slopes * t_next + intercepts    # (batch, n_features)
    target_trend = trend_at_T[:, 0:1]            # (batch, 1)

    return residuals, slopes, intercepts, target_trend


class DetrendedWindowMLP(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dims, forecast_len, dropout=0.2):
        super().__init__()

        # input shape = flattened residuals + slopes + intercepts
        input_dim = seq_len * n_features + 2 * n_features

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, forecast_len))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        residuals, slopes, intercepts, target_trend = detrend_batch(x)
        x_flat = residuals.flatten(start_dim=1)                  # (batch, seq_len * n_features)
        x_in = torch.cat([x_flat, slopes, intercepts], dim=1)    # (batch, input_dim)
        residual_pred = self.net(x_in)          # (batch, forecast_len)
        return residual_pred + target_trend     # (batch, forecast_len)
