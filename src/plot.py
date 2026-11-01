"""
Plot the learning curves of several MLflow runs side by side and print a summary table.
Output: figures/comparison.png

Usage:
    python -m src.plot           
"""

import argparse
import os

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

COMPARISON_RUNS = ["baseline", "no_augment", "no_dropout"]


# Per-epoch values of one metric of one run, in epoch order.
# MLflow does not guarantee the order of the history, so sort it by step (= epoch) first.
def metric_values(client: MlflowClient, run_id: str, key: str):
    history = sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)
    return [m.value for m in history]


# The latest finished run with this name, or None if there is none
def find_run(client: MlflowClient, run_name: str):
    exp = client.get_experiment_by_name("image-classifier")
    found = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return found[0] if found else None


# One column per run: loss on the top row, accuracy on the bottom row. Each panel shows
# the train and validation curves, with the gap between them shaded. Also prints a
# summary table.
def plot_comparison(run_names: list[str]):
    client = MlflowClient()
    metric_names = ["train_loss", "val_loss", "train_acc", "val_acc"]

    # Step 1: fetch the curves of every run, skipping runs that are not found
    experiments = []   # each item: (run name, run, {metric name: per-epoch values})
    for run_name in run_names:
        run = find_run(client, run_name)
        if run is None:
            print(f"No finished run named '{run_name}', skipping.")
            continue
        curves = {key: metric_values(client, run.info.run_id, key) for key in metric_names}
        experiments.append((run_name, run, curves))

    if not experiments:
        raise SystemExit("No runs to compare. Train them first with src/train.py.")

    # Step 2: plot a 2 x N grid (N = number of runs found).
    # sharey="row" puts each row on the same y scale, so the runs can be compared directly.
    # squeeze=False keeps axes 2-D even for a single run, so axes[row][col] always works.
    figure, axes = plt.subplots(
        2, len(experiments), figsize=(4.5 * len(experiments), 7),
        sharex=True, sharey="row", squeeze=False,
    )

    for col, (run_name, run, curves) in enumerate(experiments):
        epochs = list(range(1, len(curves["val_loss"]) + 1))

        # Row 0 plots loss, row 1 accuracy, in exactly the same way
        for row, metric in enumerate(["loss", "acc"]):
            ax = axes[row][col]
            train_curve = curves[f"train_{metric}"]
            val_curve = curves[f"val_{metric}"]

            ax.plot(epochs, train_curve, label="train")
            ax.plot(epochs, val_curve, label="val")
            # Shade the gap between train and validation
            ax.fill_between(epochs, train_curve, val_curve, color="grey", alpha=0.2)

        # test_acc is logged once at the end of training, so run.data.metrics holds it
        test_acc = run.data.metrics.get("test_acc", float("nan"))
        axes[0][col].set_title(f"{run_name} (test_acc = {test_acc:.4f})", fontweight="bold")
        axes[1][col].set_xlabel("Epoch", fontweight="bold")

    # Y-axis labels only on the leftmost column
    axes[0][0].set_ylabel("Loss", fontweight="bold")
    axes[1][0].set_ylabel("Accuracy", fontweight="bold")
    # Every panel has the same two lines, so one shared legend below the figure is enough
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    figure.suptitle("Train vs validation", fontweight="bold", fontsize=20)

    os.makedirs("figures", exist_ok=True)
    figure.savefig("figures/comparison.png", dpi=300, bbox_inches="tight")
    print("Saved figures/comparison.png\n")

    # Step 3: print a Markdown table that can be pasted into the README.
    # Best epoch: the epoch with the highest validation accuracy.
    # Final gap: train_acc - val_acc at the last epoch.
    print("| Run | Best val acc | Best epoch | Test acc | Final gap |")
    print("|---|---|---|---|---|")
    for run_name, run, curves in experiments:
        best_val_acc = run.data.metrics.get("best_val_acc", float("nan"))
        best_epoch = curves["val_acc"].index(max(curves["val_acc"])) + 1
        test_acc = run.data.metrics.get("test_acc", float("nan"))
        final_gap = curves["train_acc"][-1] - curves["val_acc"][-1]
        print(f"| {run_name} | {best_val_acc:.2%} | {best_epoch} | {test_acc:.2%} "
              f"| {final_gap:.2%} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", nargs="+", default=COMPARISON_RUNS, help="MLflow run names to plot"
    )
    plot_comparison(parser.parse_args().runs)
