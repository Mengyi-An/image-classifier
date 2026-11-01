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
    python -m src.export_onnx --model data/comparison/no_augment/model.pt
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
CHECK_IMAGES = 1000   # compare PyTorch and ONNX on the first 1000 test images


def load_model(model_path: str) -> tuple[ImageClassifier, dict]:
    # Like serve.py, accept both a bundled checkpoint and a bare state_dict
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    metrics = checkpoint.get("metrics", {})

    model = ImageClassifier()
    model.load_state_dict(state_dict)
    model.eval()   # must export in eval mode, or dropout and BatchNorm keep training behavior
    return model, metrics


def export(model: ImageClassifier, onnx_path: str) -> None:
    # torch.onnx.export runs the model once on a sample input and records every operation
    # into an ONNX graph
    dummy = torch.zeros(1, 3, *IMAGE_SIZE)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        # A variable batch dimension, so one call can classify one image or many
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def write_meta(metrics: dict, meta_path: str) -> None:
    # The web page reads its preprocessing values from this file instead of hard-coding
    # them in JavaScript, so the two can never drift apart
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
    # The first test image of each class, not hand-picked, so some may be misclassified
    os.makedirs(samples_dir, exist_ok=True)
    raw_test = datasets.CIFAR10(DATA_DIR, train=False, download=True)   # no transform: raw PIL images
    saved = set()
    for image, label in raw_test:
        name = ImageClassifier.CLASSES[label]
        if name not in saved:
            image.save(os.path.join(samples_dir, f"{name}.png"))
            saved.add(name)
        if len(saved) == len(ImageClassifier.CLASSES):
            break


def check(model: ImageClassifier, onnx_path: str) -> None:
    # Run the same images through PyTorch and ONNX Runtime and compare outputs and classes
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
