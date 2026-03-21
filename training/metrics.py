import numpy as np
import torch


def get_predictions(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            preds.append(y_pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def compute_metrics(preds, targets):
    mae = np.mean(np.abs(preds - targets))
    mse = np.mean(np.square(preds - targets))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((preds - targets) / (targets + 1e-8))) * 100
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}
