"""
Compare ablation runs from MLflow on one figure and print a summary table.

For each run name it takes the latest finished run with that name.

Usage:
    python -m src.plot_ablation
    python -m src.plot_ablation --runs baseline no_augment no_dropout no_augment_no_dropout
"""

import argparse
import os

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

DEFAULT_RUNS = ["baseline", "no_augment", "no_dropout"]


def metric_values(client: MlflowClient, run_id: str, key: str) -> list[float]:
    history = sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)
    return [m.value for m in history]


def main(run_names: list[str]) -> None:
    # 和 plot.py 一样，去当前目录下的 mlruns（或 MLFLOW_TRACKING_URI 指向的位置）找实验
    client = MlflowClient()
    exp = client.get_experiment_by_name("image-classifier")

    fig, (ax_acc, ax_gap) = plt.subplots(1, 2, figsize=(11, 4))
    rows = []

    for name in run_names:
        # 同名的 run 可能有好几次，取最近一次跑完的
        found = client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{name}' and attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not found:
            print(f"No finished run named '{name}', skipping.")
            continue
        run = found[0]

        val_acc = metric_values(client, run.info.run_id, "val_acc")
        gap = metric_values(client, run.info.run_id, "gap")
        epochs = range(1, len(val_acc) + 1)

        ax_acc.plot(epochs, val_acc, label=name)
        ax_gap.plot(epochs, gap, label=name)

        rows.append((
            name,
            run.data.metrics.get("best_val_acc", float("nan")),
            run.data.metrics.get("test_acc", float("nan")),
            gap[-1],
        ))

    ax_acc.set_title("Validation accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(loc="lower right")

    # gap = 不增强的训练图准确率 - 验证集准确率，曲线越高说明过拟合越严重
    ax_gap.set_title("Overfitting gap (train acc - val acc)")
    ax_gap.set_xlabel("Epoch")
    ax_gap.set_ylabel("Gap")
    ax_gap.axhline(0, color="grey", linewidth=0.8)
    ax_gap.legend(loc="upper left")

    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/ablation.png", dpi=150, bbox_inches="tight")
    print("Saved figures/ablation.png\n")

    # 打印成 Markdown 表格，可以直接贴进 README
    print("| Run | Best val acc | Test acc | Final gap |")
    print("|---|---|---|---|")
    for name, best_val, test, final_gap in rows:
        print(f"| {name} | {best_val:.2%} | {test:.2%} | {final_gap:.2%} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    main(parser.parse_args().runs)
