# Detrended Window MLP for Wind Speed Forecasting

A comparison of five time-series models for hourly wind speed forecasting, with a focus on the **Detrended Window MLP (DW-MLP)** — a lightweight model that explicitly decomposes each input window into a local linear trend and a stationary residual before prediction.

The central finding is that DW-MLP achieves the best data efficiency among all tested models, degrading by only **+30.2% in MAE** when trained on 25% of the data, compared to +110.6% for LSTM and +160.0% for Transformer — while also having the **fewest parameters (30,977)** and the **smallest checkpoint (124 kB)**.

---

## Table of Contents

- [Dataset](#dataset)
- [Models](#models)
- [DW-MLP — How It Works](#dw-mlp--how-it-works)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)

---

## Dataset

Hourly meteorological observations for the year 2021 (8,760 samples, no missing values).

| Column       | Description                                       |
| ------------ | ------------------------------------------------- |
| `time`       | Hourly timestamp                                  |
| `wds`        | Raw wind speed (m/s)                              |
| `wds_scaled` | Min-max scaled wind speed — **prediction target** |
| `sp`         | Surface pressure (Pa)                             |
| `t2m`        | Temperature at 2m height (K)                      |

**Preprocessing** (`scripts/prepare_data.py`):

- Chronological 70/15/15 split → `data/splits/{train,val,test}.csv`
- Z-score normalisation of `sp` and `t2m` using train statistics only
- Cyclic encoding of `hour` (period 24), `month` (period 12), `dayofyear` (period 365) as sin/cos pairs

**Final feature vector per timestep (9 features):**
`wds_scaled, sp, t2m, sin_hour, cos_hour, sin_month, cos_month, sin_dayofyear, cos_dayofyear`

**Windowing:** 48-hour lookback (`seq_len=48`), 1-hour forecast horizon (`forecast_len=1`).

---

## Models

Five models are implemented and compared, all sharing the same input shape `(batch, 48, 9)` and output shape `(batch, 1)`.

| Model       | Architecture                        | Parameters | Checkpoint   |
| ----------- | ----------------------------------- | ---------- | ------------ |
| LSTM        | 2-layer LSTM, hidden=64             | 52,545     | 208.6 kB     |
| GRU         | 2-layer GRU, hidden=64              | 39,425     | 157.3 kB     |
| TCN         | Channels [32,64,64], kernel=3       | 49,761     | 200.3 kB     |
| Transformer | d_model=64, 4 heads, 2 layers       | 100,417    | 402.6 kB     |
| **DW-MLP**  | **Hidden [64,32], detrended input** | **30,977** | **123.7 kB** |

### LSTM (`models/lstm.py`)

Reads the sequence left to right, maintaining a hidden state and a cell state across timesteps. The final hidden state is passed through a linear layer to produce the forecast.

### GRU (`models/gru.py`)

A simplified version of LSTM with a single gated state. Fewer parameters than LSTM while achieving comparable performance.

### TCN (`models/tcn.py`)

A stack of causal dilated convolutional blocks with residual connections. Doubling dilation at each block (`[1, 2, 4]`) gives an exponentially growing receptive field without recurrence. The last timestep's representation is used for the forecast.

### Transformer (`models/transformer.py`)

Projects input features to `d_model=64`, adds sinusoidal positional encoding, then passes through a 2-layer Transformer Encoder. All timesteps attend to all others simultaneously. The last timestep's output is used for the forecast.

### DW-MLP (`models/dw_mlp.py`)

Described in detail below.

---

## DW-MLP — How It Works

Standard sequence models implicitly learn both the trend and the fluctuations of a time series from data. DW-MLP makes this decomposition **explicit and analytical**, so the model only has to learn the harder part — the residuals.

### Decomposition

For each input window of shape `(seq_len, n_features)`, a local linear trend is fit per feature using ordinary least squares:

```
t = [0, 1, ..., seq_len-1]  (centered)

slope     = Σ(t_c * x_c) / Σ(t_c²)
intercept = mean(x) - slope * mean(t)
trend     = slope * t + intercept
residual  = x - trend
```

The trend is then **extrapolated** one step ahead to produce `trend_at_T` — the analytically predicted value at the forecast step.

### Prediction

The MLP receives a concatenation of:

- Flattened residuals: `(seq_len * n_features,)` = 432 values
- Slopes: `(n_features,)` = 9 values
- Intercepts: `(n_features,)` = 9 values
- **Total input: 450 values**

The MLP predicts the **residual** at the forecast step. The final output adds the extrapolated trend back:

```
ŷ = MLP(residuals, slopes, intercepts) + trend_at_T
```

### Why This Helps With Limited Data

The trend is estimated analytically — no data-driven learning is needed for the low-frequency component of the signal. The MLP only has to learn residual patterns, which are more stationary and require less data to generalise from. This is the mechanism behind DW-MLP's data efficiency advantage.

### Architecture

```
Input window x: (batch, 48, 9)
        │
   detrend_batch()
        │
   ┌────┴─────────────────┐
residuals            slopes, intercepts
(batch, 48, 9)       (batch, 9) each
   │                      │
flatten                   │
(batch, 432)              │
   └──────── cat ─────────┘
              │
       (batch, 450)
              │
       Linear(450→64) + ReLU + Dropout
              │
       Linear(64→32)  + ReLU + Dropout
              │
       Linear(32→1)
              │
       residual_pred + trend_at_T
              │
          ŷ: (batch, 1)
```

---

## Results

All models trained with: Adam optimiser, lr=1e-3, MSE loss, patience=7, max 50 epochs. Evaluated on the held-out test set (15% of data, chronologically last).

### Full data (100% of training set)

| Model       | MAE     | RMSE    | MAPE   |
| ----------- | ------- | ------- | ------ |
| LSTM        | 0.04032 | 0.05411 | 24.16% |
| GRU         | 0.04047 | 0.05433 | 24.79% |
| TCN         | 0.04200 | 0.05611 | 26.04% |
| Transformer | 0.04710 | 0.05993 | 31.86% |
| DW-MLP      | 0.04333 | 0.05762 | 25.89% |

### Half data (50% of training set)

| Model       | MAE         | RMSE        | MAPE       |
| ----------- | ----------- | ----------- | ---------- |
| LSTM        | 0.04217     | 0.05626     | 24.97%     |
| GRU         | **0.04000** | **0.05354** | **24.21%** |
| TCN         | 0.06899     | 0.08603     | 53.65%     |
| Transformer | 0.04562     | 0.06115     | 26.51%     |
| DW-MLP      | 0.04623     | 0.06188     | 28.29%     |

### Quarter data (25% of training set)

| Model       | MAE         | RMSE        | MAPE       |
| ----------- | ----------- | ----------- | ---------- |
| LSTM        | 0.08494     | 0.10343     | 66.76%     |
| GRU         | 0.05505     | 0.07068     | 34.11%     |
| TCN         | 0.06254     | 0.08045     | 38.33%     |
| Transformer | 0.12246     | 0.14924     | 80.95%     |
| **DW-MLP**  | **0.05643** | **0.07377** | **36.27%** |

### MAE degradation: full → quarter

| Model       | Degradation |
| ----------- | ----------- |
| Transformer | +160.0%     |
| LSTM        | +110.6%     |
| TCN         | +48.9%      |
| GRU         | +36.0%      |
| **DW-MLP**  | **+30.2%**  |

DW-MLP degrades the least of all models under data scarcity, while also being the smallest model by parameter count and checkpoint size.

---

## Project Structure

```
windspeed_prediction/
│
├── data/
│   ├── raw/                        # original unmodified CSVs
│   │   ├── feature_data.csv
│   │   └── target_data.csv
│   ├── processed/
│   │   └── combined_data.csv       # merged features + target
│   └── splits/                     # output of scripts/prepare_data.py
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── data_pipeline/
│   ├── __init__.py
│   ├── dataloader.py               # WindDataset and get_dataloader()
│   └── preprocessing.py            # split, normalise, cyclic encode
│
├── models/
│   ├── __init__.py
│   ├── lstm.py
│   ├── gru.py
│   ├── tcn.py
│   ├── transformer.py
│   └── dw_mlp.py                  # DetrendendWindowMLP + detrend_batch()
│
├── training/
│   ├── __init__.py
│   └── trainer.py                 # train_one_epoch, evaluate, train
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                 # get_predictions, compute_metrics
│
├── scripts/
│   ├── prepare_data.py            # run once to generate data/splits/
│   ├── train.py                   # main training entry point
│   └── plot_results.py            # generate all plots from saved results
│
├── notebooks/
│   └── eda.ipynb                  # exploratory data analysis
│
├── checkpoints/                   # saved model weights, organised by experiment
│   ├── full/
│   ├── half/
│   └── quarter/
│
├── _results/
│   ├── raw/                       # per-model JSON results (metrics + history)
│   │   ├── full/
│   │   ├── half/
│   │   └── quarter/
│   └── plots/                     # generated plots
│       ├── full/
│       ├── half/
│       └── quarter/
│
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/vinayak-ktp/detrended-window-mlp.git
cd detrended-window-mlp

# create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
pandas
matplotlib
torch
scipy
jupyter
```

---

## Usage

### 1. Prepare data

Merges raw files, splits chronologically, normalises, and encodes cyclic features:

```bash
python -m scripts.prepare_data
```

Output: `data/splits/train.csv`, `val.csv`, `test.csv`

### 2. Train a single model

```bash
# train one model on full data
python -m scripts.train --model lstm --exp full

# available models: lstm, gru, tcn, transformer, dw_mlp
# --exp sets the subdirectory name under checkpoints/ and _results/raw/
```

### 3. Train all models

```bash
python -m scripts.train --model all --exp full
```

### 4. Train on a fraction of data

```bash
python -m scripts.train --model all --exp quarter --data_frac 0.25
python -m scripts.train --model all --exp half    --data_frac 0.5
```

### 5. Generate plots

```bash
python -m scripts.plot_results
```

Reads all JSON files from `_results/raw/` and writes plots to `_results/plots/`.

### 6. Exploratory data analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## Hyperparameters

| Parameter       | Value                 |
| --------------- | --------------------- |
| `seq_len`       | 48 (48-hour lookback) |
| `forecast_len`  | 1 (1-hour ahead)      |
| `batch_size`    | 32                    |
| `num_epochs`    | 50                    |
| `patience`      | 7                     |
| `learning_rate` | 1e-3                  |
| Optimiser       | Adam                  |
| Loss            | MSELoss               |

Per-model architecture configs are defined in `scripts/train.py` under `MODEL_CONFIGS`.
