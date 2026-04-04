import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data_pipeline.dataloader import get_dataloader
from models.dw_mlp import DetrendedWindowMLP
from models.gru import GRUModel
from models.lstm import LSTMModel
from models.tcn import TCNModel
from models.transformer import TransformerModel
from training.metrics import compute_metrics, get_predictions
from training.trainer import train

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    choices=['lstm', 'gru', 'tcn', 'transformer', 'dw_mlp', 'all'])
parser.add_argument('--exp', type=str, required=True)
parser.add_argument('--n_seeds', type=int, default=5)
parser.add_argument('--seeds', type=int, nargs='+', default=None)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--data_frac', type=float, default=1.0)
args = parser.parse_args()

SEEDS = args.seeds if args.seeds is not None else list(range(args.n_seeds))

SEQ_LEN = 48
FORECAST_LEN = 1
BATCH_SIZE = 32
NUM_EPOCHS = args.epochs
PATIENCE = 7
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CONFIGS = {
    'lstm': {
        'hidden_dim': 64,
        'num_layers': 1,
        'dropout': 0.0,
    },
    'gru': {
        'hidden_dim': 64,
        'num_layers': 1,
        'dropout': 0.0,
    },
    'tcn': {
        'num_channels': [32, 64, 64],
        'kernel_size': 3,
        'dropout': 0.2,
    },
    'transformer': {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 1,
        'dropout': 0.0,
    },
    'dw_mlp': {
        'hidden_dims': [16, 8],
        'dropout': 0.0,
    },
}

TRAIN_PATH = "data/splits/train.csv"
VAL_PATH = "data/splits/val.csv"
TEST_PATH = "data/splits/test.csv"
RESULTS_DIR = os.path.join("_results", "raw", args.exp)

os.makedirs(RESULTS_DIR, exist_ok=True)

models_to_run = list(MODEL_CONFIGS.keys()) if args.model == 'all' else [args.model]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name, cfg):
    if model_name == 'lstm':
        return LSTMModel(
            n_features=9,
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            forecast_len=FORECAST_LEN,
            dropout=cfg['dropout'],
        )
    elif model_name == 'gru':
        return GRUModel(
            n_features=9,
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            forecast_len=FORECAST_LEN,
            dropout=cfg['dropout'],
        )
    elif model_name == 'tcn':
        return TCNModel(
            n_features=9,
            num_channels=cfg['num_channels'],
            kernel_size=cfg['kernel_size'],
            forecast_len=FORECAST_LEN,
            dropout=cfg['dropout'],
        )
    elif model_name == 'transformer':
        return TransformerModel(
            n_features=9,
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=cfg['num_layers'],
            forecast_len=FORECAST_LEN,
            dropout=cfg['dropout'],
        )
    elif model_name == 'dw_mlp':
        return DetrendedWindowMLP(
            seq_len=SEQ_LEN,
            n_features=9,
            hidden_dims=cfg['hidden_dims'],
            forecast_len=FORECAST_LEN,
            dropout=cfg['dropout'],
        )


summary = {}

for model_name in models_to_run:
    print(f"\nBenchmarking {model_name.upper()} ({len(SEEDS)} seeds)")
    cfg = MODEL_CONFIGS[model_name]
    all_metrics = []

    for seed in SEEDS:
        set_seed(seed)

        train_ds, _ = get_dataloader(TRAIN_PATH, SEQ_LEN, FORECAST_LEN, BATCH_SIZE, shuffle=True)
        if args.data_frac < 1.0:
            n = int(len(train_ds) * args.data_frac)
            train_ds = Subset(train_ds, range(n))
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, worker_init_fn=lambda _: np.random.seed(seed))
        _, val_dl = get_dataloader(VAL_PATH, SEQ_LEN, FORECAST_LEN, BATCH_SIZE)
        _, test_dl = get_dataloader(TEST_PATH, SEQ_LEN, FORECAST_LEN, BATCH_SIZE)

        model = build_model(model_name, cfg).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        checkpoint = os.path.join("checkpoints", args.exp, f"{model_name}_seed{seed}.pt")
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)

        print(f"\nseed={seed}")
        history = train(
            model, train_dl, val_dl, criterion, optimizer,
            device=DEVICE, num_epochs=NUM_EPOCHS, save_path=checkpoint, patience=PATIENCE,
        )

        preds, targets = get_predictions(model, test_dl, device=DEVICE)
        metrics = compute_metrics(preds, targets)
        all_metrics.append({k: float(v) for k, v in metrics.items()})

        print(f"MAE={metrics['MAE']:.4f}   MSE={metrics['MSE']:.4f}   "
              f"RMSE={metrics['RMSE']:.4f}   MAPE={metrics['MAPE']:.2f}%")

    metric_keys = list(all_metrics[0].keys())
    agg = {}
    for k in metric_keys:
        vals = [m[k] for m in all_metrics]
        agg[k] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'values': vals,
        }

    summary[model_name] = agg

    print(f"\nSummary ({len(SEEDS)} seeds):")
    for k, v in agg.items():
        print(f"{k:>4}: {v['mean']:.4f} ± {v['std']:.4f} "
              f"[min={v['min']:.4f}, max={v['max']:.4f}]")

    result_path = os.path.join(RESULTS_DIR, f"{model_name}.json")
    with open(result_path, 'w') as f:
        json.dump({
            'model': model_name,
            'seeds': SEEDS,
            'hyperparameters': {
                'seq_len': SEQ_LEN,
                'forecast_len': FORECAST_LEN,
                'batch_size': BATCH_SIZE,
                'num_epochs': NUM_EPOCHS,
                'patience': PATIENCE,
                'lr': LR,
                **cfg,
            },
            'per_seed_metrics': all_metrics,
            'aggregated': agg,
        }, f, indent=2)

summary_path = os.path.join(RESULTS_DIR, "summary.json")
with open(summary_path, 'w') as f:
    json.dump({'seeds': SEEDS, 'models': summary}, f, indent=2)
