# Detrended Window MLP for Wind Speed Forecasting

A comparison of five time-series models for hourly wind speed forecasting, with a focus on the **Detrended Window MLP (DW-MLP)** — a lightweight model that explicitly decomposes each input window into a local linear trend and a stationary residual before prediction.

Benchmarked across **10 random seeds** to produce statistically reliable comparisons, the results show that LSTM and GRU are the strongest overall performers. DW-MLP's contribution is not raw accuracy — it is **parameter efficiency and training stability**: DW-MLP achieves competitive MAPE (27.7%) with only **7,361 parameters** and a **31.4 kB checkpoint**, making it the smallest model by a large margin.

---

## Table of Contents

- [Dataset](#dataset)
- [Models](#models)
- [DW-MLP — How It Works](#dw-mlp--how-it-works)
- [Benchmarking](#benchmarking)
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

### LSTM (`models/lstm.py`)

Reads the sequence left to right, maintaining a hidden state and cell state across timesteps. The final hidden state is passed through a linear layer to produce the forecast.

### GRU (`models/gru.py`)

A simplified version of LSTM with a single gated state. Fewer parameters than LSTM while achieving comparable performance.

### TCN (`models/tcn.py`)

A stack of causal dilated convolutional blocks with residual connections. Doubling dilation at each block gives an exponentially growing receptive field without recurrence. The last timestep's representation is used for the forecast.

### Transformer (`models/transformer.py`)

Projects input features to `d_model=64`, adds sinusoidal positional encoding, then passes through a Transformer Encoder. All timesteps attend to all others simultaneously. The last timestep's output is used for the forecast.

### DW-MLP (`models/dw_mlp.py`)

Described in detail below.

---

## DW-MLP — How It Works

Standard sequence models implicitly learn both the trend and the fluctuations of a time series from data. DW-MLP makes this decomposition **explicit and analytical**, so the model only needs to learn the harder part — the residuals.

### Decomposition

For each input window of shape `(seq_len, n_features)`, a local linear trend is fit per feature using ordinary least squares:

```
t         = [0, 1, ..., seq_len-1]  (centered around mean)
slope     = Σ(t_c * x_c) / Σ(t_c²)
intercept = mean(x) - slope * mean(t)
trend     = slope * t + intercept
residual  = x - trend
```

The trend is then **extrapolated** one step ahead to produce `trend_at_T` — the analytically predicted value at the forecast step. No parameters are learned for this step.

### Prediction and Reconstruction

The MLP receives a concatenation of:

- Flattened residuals: `(seq_len * n_features,)` = 432 values
- Slopes: `(n_features,)` = 9 values
- Intercepts: `(n_features,)` = 9 values — **total input: 450 values**

The MLP predicts the **residual** at the forecast step. The final output adds the extrapolated trend back:

```
ŷ = MLP(residuals, slopes, intercepts) + trend_at_T
```

Passing slopes and intercepts alongside the residuals gives the MLP context about the local trend — two windows with identical residuals but different trends represent different physical situations.

### Architecture

```
Input window x: (batch, 48, 9)
        │
   detrend_batch() — OLS per feature, no learned params
        │
   ┌────┴──────────────────────┐
residuals                 slopes, intercepts
(batch, 48, 9)            (batch, 9) each
   │                           │
flatten                        │
(batch, 432)                   │
   └──────────── cat ──────────┘
                  │
           (batch, 450)
                  │
           Linear(450 → h₁) + ReLU + Dropout
                  │
           Linear(h₁ → h₂)  + ReLU + Dropout
                  │
           Linear(h₂ → 1)
                  │
      + trend_at_T (extrapolated, index 0 = wds)
                  │
            ŷ: (batch, 1)
```

---

## Benchmarking

All models were benchmarked across **10 random seeds** to produce stable mean and standard deviation estimates.

| Model           | Architecture        |
| --------------- | ------------------- |
| **TCN**         | channels=`[32, 64]` |
| **DW-MLP**      | hidden=`[16, 8]`    |
| **LSTM**        | hidden=64, 1 layer  |
| **GRU**         | hidden=64, 1 layer  |
| **Transformer** | d_model=64, 1 layer |

**Training protocol:** Adam optimiser, lr=1e-3, MSE loss, patience=7, max 60 epochs, batch size 32. The best validation checkpoint is restored before test evaluation.

---

## Results

| Model       | MAE         | ±std    | RMSE        | ±std    | MAPE (%)  | ±std     | Params    | Size        |
| ----------- | ----------- | ------- | ----------- | ------- | --------- | -------- | --------- | ----------- |
| LSTM        | 0.04086     | 0.00046 | 0.05451     | 0.00046 | 26.52     | 1.41     | 19,265    | 77.3 kB     |
| **GRU**     | **0.04080** | 0.00084 | **0.05442** | 0.00080 | **26.04** | 1.84     | 14,465    | 58.5 kB     |
| TCN         | 0.04394     | 0.00206 | 0.05784     | 0.00203 | 29.36     | 2.74     | 25,057    | 103.4 kB    |
| Transformer | 0.04334     | 0.00237 | 0.05753     | 0.00319 | 28.20     | 1.92     | 66,945    | 267.9 kB    |
| DW-MLP      | 0.04501     | 0.00153 | 0.06026     | 0.00213 | 27.67     | **0.83** | **7,361** | **31.4 kB** |

### Key observations

**LSTM and GRU are the best overall performers.** Both achieve MAE ~0.041 and RMSE ~0.054 consistently across all 10 seeds. Their low variance (CV ~1–2%) makes them the most reliable models on this task.

**DW-MLP has the lowest MAPE variance of all models.** DW-MLP's MAPE standard deviation is the smallest (0.83%). Its predictions are the most consistent across random seeds — a property that matters in production systems where reliability is as important as average accuracy.

**DW-MLP is the most parameter-efficient model.** With just 7,361 parameters and a 31.4 kB checkpoint — less than half the size of GRU — DW-MLP achieves competitive MAPE (27.7% vs GRU's 26.0%) at a fraction of the model footprint. The trend is handled analytically, so the learned component can be very shallow.

**TCN and Transformer are the least stable.** Both show coefficient of variation >4% on MAE across seeds, meaning their performance is sensitive to initialisation. This makes them harder to deploy reliably without extensive tuning.

### Stability summary (Coefficient of Variation on MAE, 10 seeds)

| Model       | CV (%) |
| ----------- | ------ |
| LSTM        | 1.11   |
| GRU         | 2.06   |
| DW-MLP      | 3.41   |
| TCN         | 4.69   |
| Transformer | 5.46   |

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
│   └── preprocessing.py           # split, normalise, cyclic encode
│
├── models/
│   ├── __init__.py
│   ├── lstm.py
│   ├── gru.py
│   ├── tcn.py
│   ├── transformer.py
│   └── dw_mlp.py                  # DetrendedWindowMLP + detrend_batch()
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                 # train_one_epoch, evaluate, train
│   └── metrics.py                 # get_predictions, compute_metrics
│
├── scripts/
│   ├── prepare_data.py            # run once to generate data/splits/
│   ├── train.py                   # single-run training entry point
│   └── plot_results.py            # generate plots from saved results
│
├── notebooks/
│   └── eda.ipynb                  # exploratory data analysis
│
├── checkpoints/                   # saved model weights per run and seed
│   └── run/
│       ├── lstm_seed0.pt
│       └── ...
│
├── _results/
│   ├── raw/                       # JSON results per model per run
│   │   └── run/
│   │       ├── lstm.json          # per_seed_metrics + aggregated stats
│   │       ├── summary.json       # all models aggregated in one file
│   │       └── ...
│   └── plots/                     # generated plots
│       └── run/
│
├── benchmark.py                   # multi-seed benchmarking entry point
├── requirements.txt
└── README.md
```

### Result file format

Each `{model}.json` under `_results/raw/{run}/` contains:

```json
{
  "model": "dw_mlp",
  "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "hyperparameters": { "seq_len": 48, "hidden_dims": [16, 8], "..." },
  "per_seed_metrics": [
    { "MAE": 0.043, "MSE": 0.002, "RMSE": 0.058, "MAPE": 27.1 },
    "..."
  ],
  "aggregated": {
    "MAE": { "mean": 0.045, "std": 0.0015, "min": 0.042, "max": 0.048, "values": [...] }
  }
}
```

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/vinayak-ktp/detrended-window-mlp.git
cd detrended-window-mlp

# create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# install dependencies (hardware-specific)
python install.py
```

`dependencies`:

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

```bash
python -m scripts.prepare_data
```

Output: `data/splits/train.csv`, `val.csv`, `test.csv`

### 2. Run benchmark (multi-seed)

Train all models over 10 seeds and save aggregated results:

```bash
python -m benchmark --model all --exp run
```

Train a single model:

```bash
python -m benchmark --model dw_mlp --exp run
```

With a custom number of seeds or specific seeds:

```bash
python -m benchmark --model all --exp run --n_seeds 5
python -m benchmark --model all --exp run --seeds 0 1 2 3 4
```

On a fraction of training data:

```bash
python -m benchmark --model all --exp run_quarter --data_frac 0.25
```

### 3. Train a single model (single run, no multi-seed)

```bash
python -m scripts.train --model lstm --exp run
```

### 4. Generate plots

```bash
python -m scripts.plot_results
```

### 5. Exploratory data analysis

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
| `num_epochs`    | 60                    |
| `patience`      | 7                     |
| `learning_rate` | 1e-3                  |
| Optimiser       | Adam                  |
| Loss            | MSELoss               |
| Seeds           | 0–9 (10 seeds)        |

### Per-model architecture configs

| Model       | Configuration                              |
| ----------- | ------------------------------------------ |
| LSTM        | hidden=64, layers=1, dropout=0.0           |
| GRU         | hidden=64, layers=1, dropout=0.0           |
| TCN         | channels=[32,64], kernel=3, dropout=0.2    |
| Transformer | d_model=64, nhead=4, layers=1, dropout=0.0 |
| DW-MLP      | hidden=[16,8], dropout=0.1                 |
