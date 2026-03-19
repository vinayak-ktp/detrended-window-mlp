import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim

from data_pipeline.dataloader import get_dataloader
from evaluation.metrics import compute_metrics, get_predictions
from models.dw_mlp import DetrendedWindowMLP
from models.gru import GRUModel
from models.lstm import LSTMModel
from models.tcn import TCNModel
from models.transformer import TransformerModel
from training.trainer import train

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    choices=['lstm', 'gru', 'tcn', 'transformer', 'dw_mlp'])
args = parser.parse_args()

SEQ_LEN = 48
FORECAST_LEN = 1
BATCH_SIZE = 32
NUM_EPOCHS = 50
PATIENCE = 10
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CONFIGS = {
    'lstm': {
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.2,
    },
    'gru': {
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.2,
    },
    'tcn': {
        'num_channels': [32, 64, 64],
        'kernel_size': 3,
        'dropout': 0.2,
    },
    'transformer': {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.1,
    },
    'dw_mlp': {
        'hidden_dims': [256, 128, 64],
        'dropout': 0.2,
    },
}

TRAIN_PATH = "data/splits/train.csv"
VAL_PATH = "data/splits/val.csv"
TEST_PATH = "data/splits/test.csv"
CHECKPOINT = f"checkpoints/{args.model}.pt"
RESULTS_DIR = "_results"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

_, train_dl = get_dataloader(
    TRAIN_PATH,
    SEQ_LEN,
    FORECAST_LEN,
    BATCH_SIZE,
    shuffle=True,
)

_, val_dl = get_dataloader(
    VAL_PATH,
    SEQ_LEN,
    FORECAST_LEN,
    BATCH_SIZE,
)

_, test_dl = get_dataloader(
    TEST_PATH,
    SEQ_LEN,
    FORECAST_LEN,
    BATCH_SIZE,
)

_, plot_dl = get_dataloader(
    TEST_PATH,
    SEQ_LEN,
    FORECAST_LEN,
    BATCH_SIZE,
    stride=FORECAST_LEN
)

cfg = MODEL_CONFIGS[args.model]

if args.model == 'lstm':
    model = LSTMModel(
        n_features=9,
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        forecast_len=FORECAST_LEN,
        dropout=cfg['dropout'],
    ).to(DEVICE)
elif args.model == 'gru':
    model = GRUModel(
        n_features=9,
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        forecast_len=FORECAST_LEN,
        dropout=cfg['dropout'],
    ).to(DEVICE)
elif args.model == 'tcn':
    model = TCNModel(
        n_features=9,
        num_channels=cfg['num_channels'],
        kernel_size=cfg['kernel_size'],
        forecast_len=FORECAST_LEN,
        dropout=cfg['dropout'],
    ).to(DEVICE)
elif args.model == 'transformer':
    model = TransformerModel(
        n_features=9,
        d_model=cfg['d_model'],
        nhead=cfg['nhead'],
        num_layers=cfg['num_layers'],
        forecast_len=FORECAST_LEN,
        dropout=cfg['dropout'],
    ).to(DEVICE)
elif args.model == 'dw_mlp':
    model = DetrendedWindowMLP(
        seq_len=SEQ_LEN,
        n_features=9,
        hidden_dims=cfg['hidden_dims'],
        forecast_len=FORECAST_LEN,
        dropout=cfg['dropout'],
    ).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

history = train(
    model,
    train_dl,
    val_dl,
    criterion,
    optimizer,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    save_path=CHECKPOINT,
    patience=PATIENCE,
)

preds, targets = get_predictions(model, test_dl, device=DEVICE)
metrics = compute_metrics(preds, targets)

preds_plot, targets_plot = get_predictions(model, plot_dl, device=DEVICE)

print("Test metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

results = {
    "history": history,
    "metrics": {k: float(v) for k, v in metrics.items()},
    "predictions": preds_plot.tolist(),
    "targets": targets_plot.tolist(),
}

with open(os.path.join(RESULTS_DIR, f"{args.model}.json"), "w") as f:
    json.dump(results, f)
