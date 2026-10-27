"""
Confusion matrix of a trained checkpoint on the CIFAR-10 test set.

Saves figures/confusion_matrix.png and prints per-class accuracy and the
most confused class pairs.

Usage:
    python -m src.confusion
    python -m src.confusion --model data/ablation/no_augment/model.pt
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from src.model import ImageClassifier
from src.train import DATA_DIR, DEVICE, EVAL_TRANSFORM, MODEL_PATH

TOP_PAIRS = 5


def main(model_path: str) -> None:
    # 和 serve.py 一样，兼容"打包的 checkpoint"和"只有权重"两种格式
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)

    model = ImageClassifier().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=EVAL_TRANSFORM)
    loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    # 收集所有测试图的真实标签和预测标签
    labels, preds = [], []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(DEVICE))
            preds.extend(outputs.argmax(1).cpu().tolist())
            labels.extend(targets.tolist())

    classes = ImageClassifier.CLASSES

    # normalize="true" 按行归一化：每一行是一个真实类别，格子里是"被预测成某类的比例"，
    # 对角线就是每类的准确率
    cm = confusion_matrix(labels, preds, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, cmap="Blues", values_format=".2f", xticks_rotation=45, colorbar=False
    )
    accuracy = np.mean(np.array(labels) == np.array(preds))
    ax.set_title(f"Confusion matrix on test set (accuracy {accuracy:.2%})")

    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved figures/confusion_matrix.png\n")

    print("Per-class accuracy (worst first):")
    for i in np.argsort(cm.diagonal()):
        print(f"  {classes[i]:<10} {cm[i, i]:.2%}")

    # 把对角线清零，剩下的最大值就是最常见的错误：真实是 A，被预测成 B
    errors = cm.copy()
    np.fill_diagonal(errors, 0)
    print(f"\nTop {TOP_PAIRS} confusions (true -> predicted, share of the true class):")
    for flat in np.argsort(errors, axis=None)[::-1][:TOP_PAIRS]:
        true_i, pred_i = np.unravel_index(flat, errors.shape)
        print(f"  {classes[true_i]:<10} -> {classes[pred_i]:<10} {errors[true_i, pred_i]:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    main(parser.parse_args().model)
