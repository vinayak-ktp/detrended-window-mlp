import torch
import torch.nn as nn


def detrend_batch(x, forecast_len=1):
    batch, seq_len, n_features = x.shape

    t = torch.arange(seq_len, dtype=torch.float32, device=x.device)
    t_mean = t.mean()
    t_centered = t - t_mean
    t_3d = t_centered.reshape(1, seq_len, 1)

    y_mean = x.mean(dim=1)

    y_mean_3d = y_mean.reshape(batch, 1, n_features)
    x_centered = x - y_mean_3d

    num = (t_3d * x_centered).sum(dim=1)
    den = (t_centered ** 2).sum()

    slopes = num / den                        # (batch, n_features)
    intercepts = y_mean - slopes * t_mean     # (batch, n_features)

    slopes_3d = slopes.reshape(batch, 1, n_features)
    intercepts_3d = intercepts.reshape(batch, 1, n_features)

    trend = t_3d * slopes_3d + intercepts_3d
    residuals = x - trend

    # extrapolate for all forecast steps
    t_forecast = torch.arange(seq_len, seq_len + forecast_len,
                               dtype=torch.float32, device=x.device) - t_mean   # (forecast_len,)
    t_forecast_2d = t_forecast.reshape(1, forecast_len)                         # (1, forecast_len)

    # target feature only (index 0 = wds)
    target_slope = slopes[:, 0:1]           # (batch, 1)
    target_intercept = intercepts[:, 0:1]   # (batch, 1)
    target_trend = target_slope * t_forecast_2d + target_intercept  # (batch, forecast_len)

    return residuals, slopes, intercepts, target_trend


class DetrendedWindowMLP(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dims, forecast_len, dropout=0.2):
        super().__init__()
        self.forecast_len = forecast_len

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
        residuals, slopes, intercepts, target_trend = detrend_batch(x, self.forecast_len)
        x_flat = residuals.flatten(start_dim=1)                  # (batch, seq_len * n_features)
        x_in = torch.cat([x_flat, slopes, intercepts], dim=1)    # (batch, input_dim)
        residual_pred = self.net(x_in)          # (batch, forecast_len)
        return residual_pred + target_trend     # (batch, forecast_len)
