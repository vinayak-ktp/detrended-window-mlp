import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True)
args = parser.parse_args()

SEQ_LEN = 48
FORECAST_LEN = 1
RESULTS_DIR = os.path.join("_results", "raw", args.exp)
PER_MODEL_DIR = os.path.join("_results", "plots", args.exp, "per_model")
COMPARISON_DIR = os.path.join("_results", "plots", args.exp, "comparison")
MODELS = ["lstm", "gru", "tcn", "transformer", "dw_mlp"]
MODEL_COLORS = {
    "lstm": "#4e79a7",
    "gru": "#f28e2b",
    "tcn": "#e15759",
    "transformer": "#76b7b2",
    "dw_mlp": "#59a14f",
}

os.makedirs(PER_MODEL_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def is_benchmark(result):
    return "aggregated" in result


def load_results():
    data = {}
    for model in MODELS:
        path = os.path.join(RESULTS_DIR, f"{model}.json")
        if not os.path.exists(path):
            print(f"skipping {model} ({path} not found)")
            continue
        with open(path) as f:
            data[model] = json.load(f)
    return data


def load_targets(seq_len, forecast_len):
    import pandas as pd

    df = pd.read_csv("data/splits/test.csv")
    targets = df["wds_scaled"].values.astype(np.float32)
    n_windows = len(targets) - seq_len - forecast_len + 1
    result = np.array([
        targets[seq_len + i : seq_len + i + forecast_len]
        for i in range(n_windows)
    ])
    return result


# Train format
def plot_loss_curve(model_name, history, color):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, train_loss, label="Train loss", color=color, lw=2)
    ax.plot(epochs, val_loss, label="Val loss", color=color, lw=2, linestyle="--", alpha=0.7)

    best_epoch = int(np.argmin(val_loss)) + 1
    best_val = min(val_loss)
    ax.axvline(best_epoch, color="black", lw=1, linestyle=":")
    ax.scatter([best_epoch], [best_val], color="black", zorder=5,
               label=f"Best val epoch {best_epoch} ({best_val:.4f})")

    ax.set_title(f"{model_name.upper()} Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    plt.tight_layout()

    out = os.path.join(PER_MODEL_DIR, f"{model_name}_loss_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()


def plot_predictions(model_name, predictions, targets, color):
    preds = np.array(predictions).squeeze()
    trues = np.array(targets).squeeze()
    steps = np.arange(len(preds))
    errors = trues - preds

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    axes[0].plot(steps, trues, label="True", color="black", lw=1.2, alpha=0.8)
    axes[0].plot(steps, preds, label="Predicted", color=color, lw=1.0, alpha=0.85)
    axes[0].set_title(f"{model_name.upper()} - Predictions vs True (test set)")
    axes[0].set_ylabel("wds_scaled")
    axes[0].legend()

    axes[1].fill_between(steps, errors, alpha=0.4, color=color)
    axes[1].axhline(0, color="black", lw=0.8, linestyle="--")
    axes[1].set_title("Error (true − predicted)")
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Timestep")

    plt.tight_layout()
    out = os.path.join(PER_MODEL_DIR, f"{model_name}_predictions.png")
    plt.savefig(out, dpi=150)
    plt.close()


def plot_scatter(model_name, predictions, targets, color):
    preds = np.array(predictions).squeeze()
    trues = np.array(targets).squeeze()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(trues, preds, s=6, alpha=0.3, color=color)
    lims = [min(trues.min(), preds.min()), max(trues.max(), preds.max())]
    ax.plot(lims, lims, "k--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("True wds_scaled")
    ax.set_ylabel("Predicted wds_scaled")
    ax.set_title(f"{model_name.upper()} - Scatter")
    ax.legend()
    plt.tight_layout()

    out = os.path.join(PER_MODEL_DIR, f"{model_name}_scatter.png")
    plt.savefig(out, dpi=150)
    plt.close()


def plot_metrics_bar(all_results):
    metrics_to_plot = ["MAE", "RMSE", "MAPE"]
    model_names = list(all_results.keys())
    colors = [MODEL_COLORS[m] for m in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric in zip(axes, metrics_to_plot):
        values = [all_results[m]["metrics"][metric] for m in model_names]
        bars = ax.bar(model_names, values, color=colors, edgecolor="white", width=0.5)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(values) * 0.01,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x")

    plt.suptitle("Model Comparison - Test Set Metrics", fontsize=13)
    plt.tight_layout()

    out = os.path.join(COMPARISON_DIR, "metrics_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()


def plot_loss_curves_comparison(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, result in all_results.items():
        color = MODEL_COLORS[model_name]
        train_loss = result["history"]["train_loss"]
        val_loss = result["history"]["val_loss"]
        epochs = range(1, len(train_loss) + 1)

        axes[0].plot(epochs, train_loss, label=model_name, color=color, lw=1.8)
        axes[1].plot(epochs, val_loss, label=model_name, color=color, lw=1.8)

    axes[0].set_title("Train Loss - All Models")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()

    axes[1].set_title("Validation Loss - All Models")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].legend()

    plt.suptitle("Training & Validation Loss Comparison", fontsize=13)
    plt.tight_layout()

    out = os.path.join(COMPARISON_DIR, "loss_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()


def plot_predictions_comparison(all_results):
    SHOW = 200

    fig, ax = plt.subplots(figsize=(14, 5))

    first_result = next(iter(all_results.values()))
    trues = np.array(first_result["targets"]).squeeze()
    ax.plot(trues[:SHOW], label="True", color="black", lw=1.5, zorder=10)

    for model_name, result in all_results.items():
        preds = np.array(result["predictions"]).squeeze()
        ax.plot(preds[:SHOW], label=model_name,
                color=MODEL_COLORS[model_name], lw=1.0, alpha=0.75)

    ax.set_title(f"Predictions Comparison - First {SHOW} Test Timesteps")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("wds_scaled")
    ax.legend()
    plt.tight_layout()

    out = os.path.join(COMPARISON_DIR, "predictions_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()


# Benchmark format
def plot_benchmark_per_model(model_name, result, color):
    metrics_to_plot = ["MAE", "RMSE", "MAPE"]
    agg = result["aggregated"]
    seeds = result["seeds"]
    x = np.arange(len(seeds))
    width = 0.6

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, metric in zip(axes, metrics_to_plot):
        vals = agg[metric]["values"]
        mean = agg[metric]["mean"]
        bars = ax.bar(x, vals, width, color=color, alpha=0.75, edgecolor="white")
        ax.axhline(mean, color="black", lw=1.2, linestyle="--", label=f"mean={mean:.4f}")

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(vals) * 0.01,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xlabel("Seed")
        ax.set_xticks(x)
        ax.set_xticklabels(seeds)
        ax.legend(fontsize=8)

    plt.suptitle(f"{model_name.upper()} - Per-seed Metrics", fontsize=13)
    plt.tight_layout()

    out = os.path.join(PER_MODEL_DIR, f"{model_name}_per_seed.png")
    plt.savefig(out, dpi=150)
    plt.close()


def plot_benchmark_metrics_bar(all_results):
    metrics_to_plot = ["MAE", "RMSE", "MAPE"]
    model_names = list(all_results.keys())
    colors = [MODEL_COLORS[m] for m in model_names]
    x = np.arange(len(model_names))
    width = 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric in zip(axes, metrics_to_plot):
        means = [all_results[m]["aggregated"][metric]["mean"] for m in model_names]
        stds = [all_results[m]["aggregated"][metric]["std"] for m in model_names]
        bars = ax.bar(x, means, width, yerr=stds, color=colors, edgecolor="white",
                      capsize=4, error_kw={"lw": 1.5})

        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + max(means) * 0.01,
                f"{mean:.4f}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)

    n_seeds = next(iter(all_results.values())).get("seeds", [])
    plt.suptitle(f"Model Comparison - Mean ± Std ({len(n_seeds)} seeds)", fontsize=13)
    plt.tight_layout()

    out = os.path.join(COMPARISON_DIR, "metrics_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()


def plot_seed_variance(all_results):
    metrics_to_plot = ["MAE", "RMSE", "MAPE"]
    model_names = list(all_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric in zip(axes, metrics_to_plot):
        for model_name in model_names:
            vals = all_results[model_name]["aggregated"][metric]["values"]
            seeds = range(len(vals))
            ax.plot(seeds, vals, marker="o", label=model_name,
                    color=MODEL_COLORS[model_name], lw=1.5)

        ax.set_title(metric)
        ax.set_xlabel("Seed index")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)

    plt.suptitle("Per-seed Metrics by Model", fontsize=13)
    plt.tight_layout()

    out = os.path.join(COMPARISON_DIR, "seed_variance.png")
    plt.savefig(out, dpi=150)
    plt.close()


if __name__ == "__main__":
    all_results = load_results()

    if not all_results:
        print("No result files found. Train some models first.")
        exit()

    benchmark = any(is_benchmark(r) for r in all_results.values())

    if benchmark:
        print("Benchmark format (aggregated across seeds)")
        for model_name, result in all_results.items():
            plot_benchmark_per_model(model_name, result, MODEL_COLORS[model_name])
        plot_benchmark_metrics_bar(all_results)
        plot_seed_variance(all_results)
    else:
        print("Train format (single run)")
        targets = load_targets(SEQ_LEN, FORECAST_LEN)
        for result in all_results.values():
            result["targets"] = targets.tolist()

        for model_name, result in all_results.items():
            color = MODEL_COLORS[model_name]
            plot_loss_curve(model_name, result["history"], color)
            plot_predictions(model_name, result["predictions"], result["targets"], color)
            plot_scatter(model_name, result["predictions"], result["targets"], color)

        plot_metrics_bar(all_results)
        plot_loss_curves_comparison(all_results)
        plot_predictions_comparison(all_results)

    print(f"All plots saved to _results/plots/{args.exp}/")
