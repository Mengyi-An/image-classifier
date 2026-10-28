"""
Plot the learning curves of several MLflow runs side by side and print a summary table.

Runs are picked by run name; if several share a name, the latest finished one is used.
Output: figures/comparison.png

Usage:
    python -m src.plot                         # baseline vs no_augment vs no_dropout
    python -m src.plot --runs baseline         # only one run
    python -m src.plot --runs baseline no_augment no_dropout no_augment_no_dropout
"""

import argparse
import os

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

COMPARISON_RUNS = ["baseline", "no_augment", "no_dropout"]


# MLflow 不保证返回顺序，所以先按 step（也就是 epoch）排序
# 这里取 run_id的时候，对于每一个 run name，取最新一次的 run id
def metric_values(client: MlflowClient, run_id: str, key: str):

    # 取出数据并排序，.get_metric_history返回的是一个 list，但是返回的顺序不是按照 epoch=1,2,3...，所以这里要进行 sorted
    history = sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)
    return [m.value for m in history]


# 按名字找 run，只要正常跑完的；同名的有好几次时，按开始时间倒序取第 1 条 = 最近一次
# 找不到时返回 None
def find_run(client: MlflowClient, run_name: str):
    exp = client.get_experiment_by_name("image-classifier") # 按名字查找实验
    found = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return found[0] if found else None


#   train（虚线）：训练时边学边算的，增强和 dropout 开着，用来看训练有没有正常收敛
#   train (eval mode)：训练集里 5000 张原图，eval 模式下算的，和 val 条件一致
#   val：验证集
# train (eval mode) 和 val 之间涂色的区域就是过拟合的程度。最后打印一个对比表格
def plot_comparison(run_names: list[str]):
    client = MlflowClient()
    metric_names = [
        "train_loss", "train_eval_loss", "val_loss",
        "train_acc", "train_eval_acc", "val_acc",
    ]

    # 第 1 步：先把每组的曲线都取出来。找不到的、或者是旧代码跑的（没有 train_eval 指标）就跳过
    experiments = []   # 每个元素是 (run 名, run, {指标名: 每个 epoch 的值})
    for run_name in run_names:
        run = find_run(client, run_name)
        if run is None:
            print(f"No finished run named '{run_name}', skipping.")
            continue
        curves = {key: metric_values(client, run.info.run_id, key) for key in metric_names}
        if not curves["train_eval_acc"]:
            print(f"Run '{run_name}' has no train_eval metrics (old code?), skipping.")
            continue
        experiments.append((run_name, run, curves))

    if not experiments:
        raise SystemExit("No runs to compare. Train them first with src/train.py.")

    # 第 2 步：画图。2 行 × N 列（N = 找到了几组）
    # sharey="row"：同一行的子图共用纵轴刻度，这样三组的涂色区域大小才能直接比较
    # squeeze=False：只有一组时 axes 也保持二维，下面 axes[row][col] 的写法不用改
    figure, axes = plt.subplots(
        2, len(experiments), figsize=(4.5 * len(experiments), 7),
        sharex=True, sharey="row", squeeze=False,
    )

    for col, (run_name, run, curves) in enumerate(experiments):
        epochs = list(range(1, len(curves["val_loss"]) + 1))

        # row 0 画 loss，row 1 画 acc，画法完全一样，只是取的指标不同
        for row, metric in enumerate(["loss", "acc"]):
            ax = axes[row][col]
            train_curve = curves[f"train_{metric}"]
            train_eval_curve = curves[f"train_eval_{metric}"]
            val_curve = curves[f"val_{metric}"]

            # 虚线、颜色浅一点，表示它只是参考，不用来和 val 比较
            ax.plot(epochs, train_curve, label="train", linestyle="--", alpha=0.6)
            ax.plot(epochs, train_eval_curve, label="train (eval mode)")
            ax.plot(epochs, val_curve, label="val")
            # train (eval mode) 和 val 之间涂上浅灰色，这块区域就是 gap，越宽说明过拟合越严重
            ax.fill_between(epochs, train_eval_curve, val_curve, color="grey", alpha=0.2)

        axes[0][col].set_title(run_name)
        axes[1][col].set_xlabel("Epoch")

    # 纵轴名称只写在最左边一列
    axes[0][0].set_ylabel("Loss")
    axes[1][0].set_ylabel("Accuracy")
    # 每张子图的三条线都一样，所以整张图只放一个图例，放在最下面，不会挡住曲线
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    figure.suptitle("Shaded area = gap between train (eval mode) and val = overfitting")

    os.makedirs("figures", exist_ok=True)
    figure.savefig("figures/comparison.png", dpi=300, bbox_inches="tight")
    print("Saved figures/comparison.png\n")

    # 第 3 步：打印成 Markdown 表格，可以直接贴进 README
    # best_val_acc 和 test_acc 只在训练结束时记录一次，run.data.metrics 里就是那个值
    # Final gap = 最后一个 epoch 的 train_eval_acc - val_acc
    print("| Run | Best val acc | Test acc | Final gap |")
    print("|---|---|---|---|")
    for run_name, run, curves in experiments:
        best_val_acc = run.data.metrics.get("best_val_acc", float("nan"))
        test_acc = run.data.metrics.get("test_acc", float("nan"))
        final_gap = curves["train_eval_acc"][-1] - curves["val_acc"][-1]
        print(f"| {run_name} | {best_val_acc:.2%} | {test_acc:.2%} | {final_gap:.2%} |")


# 直接运行 python -m src.plot 时才画图；被别的文件 import 时不会自动执行
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", nargs="+", default=COMPARISON_RUNS, help="MLflow run names to plot"
    )
    plot_comparison(parser.parse_args().runs)
