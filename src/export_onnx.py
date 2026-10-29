"""
Export the trained model to ONNX for the browser demo in docs/.

Writes:
    docs/model.onnx        the model, input (N, 3, 32, 32) normalized images -> output (N, 10) logits
    docs/model_meta.json   class names and normalization values the web page needs for preprocessing
    docs/samples/*.png     the first test image of each class, used as clickable examples

Then checks that ONNX Runtime gives the same predictions as PyTorch on the test set.

Needs two extra packages that the API and training do not use:
    pip install onnx==1.17.0 onnxruntime
(newer onnx versions need protobuf>=6, which conflicts with mlflow 2.13)

Usage:
    python -m src.export_onnx
    python -m src.export_onnx --model data/ablation/no_augment/model.pt
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import torch
from torchvision import datasets

from src.model import IMAGE_SIZE, NORM_MEAN, NORM_STD, ImageClassifier
from src.train import DATA_DIR, EVAL_TRANSFORM

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
DEFAULT_MODEL = os.path.join(DATA_DIR, "model.pt")
CHECK_IMAGES = 1000   # 用测试集的前 1000 张图比较 PyTorch 和 ONNX 的结果


def load_model(model_path: str) -> tuple[ImageClassifier, dict]:
    # 和 serve.py 一样，兼容"打包的 checkpoint"和"只有权重"两种格式
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    metrics = checkpoint.get("metrics", {})

    model = ImageClassifier()
    model.load_state_dict(state_dict)
    model.eval()   # 导出前一定要切到 eval 模式，否则 dropout / BatchNorm 会按训练模式导出
    return model, metrics


def export(model: ImageClassifier, onnx_path: str) -> None:
    # torch.onnx.export 会用一个示例输入把模型跑一遍，记录下所有计算步骤，存成 ONNX 文件
    dummy = torch.zeros(1, 3, *IMAGE_SIZE)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        # 第 0 维（batch 大小）设成可变的，这样一次可以预测一张，也可以预测多张
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def write_meta(metrics: dict, meta_path: str) -> None:
    # 网页做预处理需要的信息都从这里读，不在 JavaScript 里再写一份，避免两边不一致
    meta = {
        "classes": ImageClassifier.CLASSES,
        "image_size": list(IMAGE_SIZE),
        "mean": list(NORM_MEAN),
        "std": list(NORM_STD),
        "test_acc": metrics.get("test_acc"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def save_samples(samples_dir: str) -> None:
    # 每个类别取测试集里出现的第一张图，不挑选，所以示例里也可能有模型认错的
    os.makedirs(samples_dir, exist_ok=True)
    raw_test = datasets.CIFAR10(DATA_DIR, train=False, download=True)   # 不做 transform，拿到原始 PIL 图片
    saved = set()
    for image, label in raw_test:
        name = ImageClassifier.CLASSES[label]
        if name not in saved:
            image.save(os.path.join(samples_dir, f"{name}.png"))
            saved.add(name)
        if len(saved) == len(ImageClassifier.CLASSES):
            break


def check(model: ImageClassifier, onnx_path: str) -> None:
    # 同一批图分别用 PyTorch 和 ONNX Runtime 预测，比较输出的数值和预测的类别
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=EVAL_TRANSFORM)
    images = torch.stack([test_set[i][0] for i in range(CHECK_IMAGES)])

    with torch.no_grad():
        torch_logits = model(images).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"input": images.numpy()})[0]

    max_diff = np.abs(torch_logits - onnx_logits).max()
    same_class = (torch_logits.argmax(1) == onnx_logits.argmax(1)).mean()
    print(f"Checked {CHECK_IMAGES} test images: max logit difference {max_diff:.2e}, "
          f"same predicted class {same_class:.2%}")
    if same_class < 1.0:
        raise SystemExit("ONNX predictions differ from PyTorch. Do not publish this model.")


def main(model_path: str) -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)
    onnx_path = os.path.join(DOCS_DIR, "model.onnx")

    model, metrics = load_model(model_path)
    export(model, onnx_path)
    write_meta(metrics, os.path.join(DOCS_DIR, "model_meta.json"))
    save_samples(os.path.join(DOCS_DIR, "samples"))
    print(f"Exported {model_path} -> {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")

    check(model, onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    main(parser.parse_args().model)
