"""
Train the CNN on CIFAR-10 and log everything to MLflow.

Pipeline: download CIFAR-10 -> augment -> train the CNN -> evaluate every epoch
-> log to MLflow -> keep the best checkpoint -> evaluate it once on the test set.

CIFAR-10 is downloaded automatically on first run (~170MB). The 50k training images are split into 45k train / 5k validation (fixed split).
The best epoch is picked on validation; the test set is used exactly once at the end.

Usage:
    python -m src.train

Comparison runs:
    python -m src.train --no-augment
    python -m src.train --no-dropout
"""

import argparse
import json
import os
import random

import mlflow
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.model import IMAGE_SIZE, NORM_MEAN, NORM_STD, ImageClassifier

# The data/ folder at the repository root (one level up from src/)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED = 42

# 5,000 of the 50,000 training images are held out for validation. The split uses its
# own fixed seed, so every run (any training seed, any comparison setting) is validated
# on exactly the same images and the results stay comparable.
VAL_SIZE = 5000
SPLIT_SEED = 0

# Training-time preprocessing with data augmentation
TRAIN_TRANSFORM = transforms.Compose([
    # Pad to 40x40 and randomly crop back to 32x32, i.e. a random shift of up to 4 pixels
    transforms.RandomCrop(IMAGE_SIZE[0], padding=4),
    # A horizontally flipped airplane, horse or truck is still the same class
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # Normalize each channel with the dataset mean and std
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# Preprocessing for the validation and test sets: no augmentation
EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def set_seed(seed: int, deterministic: bool = True) -> None:
    # Seed every source of randomness: Python, NumPy, PyTorch on CPU and on all GPUs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # GPU only: deterministic cuDNN kernels make results repeatable at some cost in speed
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# Build the train / validation / test loaders. Augmentation is applied on the fly each
# time an image is loaded. augment=False turns it off for the comparison run.
def get_dataloaders(batch_size: int, data_dir: str, augment: bool = True):
    train_transform = TRAIN_TRANSFORM if augment else EVAL_TRANSFORM

    # train=True gives the 50k training images, train=False the 10k test images.
    # The training images are loaded twice: with augmentation for training, and without
    # augmentation for validation.
    train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    clean_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=EVAL_TRANSFORM)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=EVAL_TRANSFORM)

    # Shuffle the indices with a dedicated, fixed-seed generator: the first 5,000 become
    # the validation set and the remaining 45,000 the training set. The two do not overlap.
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    indices = torch.randperm(len(train_full), generator=generator).tolist()
    val_idx, train_idx = indices[:VAL_SIZE], indices[VAL_SIZE:]

    train_set = Subset(train_full, train_idx)
    val_set = Subset(clean_full, val_idx)

    # A Dataset returns one sample at a time; a DataLoader groups samples into batches of
    # (images, labels) tensors. num_workers=2 loads data in two background processes.
    def make_loader(dataset, shuffle: bool = False):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return (
        # Reshuffle the training set every epoch so batches differ from epoch to epoch;
        # evaluation sets don't need shuffling
        make_loader(train_set, shuffle=True),
        make_loader(val_set),
        make_loader(test_set),
    )


# Average loss and accuracy of the model on one dataset (validation or test)
def evaluate(model: nn.Module, loader, criterion):
    # Eval mode: dropout is turned off and BatchNorm uses its running statistics
    model.eval()

    total_loss, correct, total = 0.0, 0, 0

    # No gradients are needed for evaluation, which saves memory and time
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            # criterion returns the mean loss of the batch; multiply by the actual batch
            # size (the last batch is usually smaller) to accumulate the total loss
            total_loss += criterion(outputs, labels).item() * len(labels)

            # The predicted class is the one with the highest score
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


# Save the model together with how it was trained and how well it did.
# out_dir is data/ for the baseline and data/comparison/<run name>/ for comparison runs,
# so they never overwrite data/model.pt, which serve.py loads.
def save_checkpoint(
    model: nn.Module, params: dict, seed: int, metrics: dict, out_dir: str = DATA_DIR
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Weights, hyperparameters, seed, metrics and class names in one file
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": params,
            "seed": seed,
            "metrics": metrics,
            "classes": ImageClassifier.CLASSES,
        },
        os.path.join(out_dir, "model.pt"),
    )

    # The same information as human-readable JSON
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"params": params, "seed": seed, "metrics": metrics}, f, indent=2)


def train(
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int = DEFAULT_SEED,
    deterministic: bool = True,
    *,  # the arguments below must be passed by name, e.g. augment=False
    augment: bool = True,
    dropout: bool = True,
    run_name: str = "baseline",
    out_dir: str = DATA_DIR,
):
    set_seed(seed, deterministic)

    data_dir = DATA_DIR
    mlflow.set_experiment("image-classifier")

    # The run configuration, logged to MLflow and stored in the checkpoint
    params = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "deterministic": deterministic,
        "device": DEVICE,
        "augment": augment,
        "dropout": dropout,
        "val_size": VAL_SIZE,
    }

    # run_name tells the baseline and the comparison runs apart in MLflow
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size, data_dir, augment
        )

        # A dropout rate of 0 disables dropout without changing the architecture
        if dropout:
            model = ImageClassifier().to(DEVICE)
        else:
            model = ImageClassifier(dropout=0.0, conv_dropout=0.0).to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        # weight_decay adds L2 regularization against overfitting
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            total_acc = 0
            total = 0  # number of images seen so far in this epoch

            # The weights are updated after every batch
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()              # clear gradients from the previous step
                outputs = model(images)            # forward pass
                loss = criterion(outputs, labels)
                loss.backward()                    # backpropagation: compute gradients
                optimizer.step()                   # update the weights

                # Running training loss and accuracy (measured with augmentation and
                # dropout active)
                total_loss += loss.item() * len(labels)
                total_acc += (outputs.argmax(1) == labels).sum().item()
                total += len(labels)

            train_loss = total_loss / total
            train_acc = total_acc / total

            scheduler.step()  # update the learning rate once per epoch

            val_loss, val_acc = evaluate(model, val_loader, criterion)

            # Per-epoch metrics; plot.py draws train and val curves from them
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch
            )

            print(f"Epoch {epoch}/{epochs}  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

            # Keep the checkpoint with the best validation accuracy
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    model,
                    params,
                    seed,
                    {"val_loss": val_loss, "val_acc": val_acc, "best_val_acc": best_acc,
                     "train_acc": train_acc, "epoch": epoch},
                    out_dir,
                )

        # Reload the best epoch and evaluate it on the test set exactly once.
        # This is the unbiased final score.
        checkpoint = torch.load(
            os.path.join(out_dir, "model.pt"), map_location=DEVICE, weights_only=True
        )
        model.load_state_dict(checkpoint["state_dict"])
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        # Add the test metrics to the saved checkpoint
        metrics = {**checkpoint["metrics"], "test_loss": test_loss, "test_acc": test_acc}
        save_checkpoint(model, params, seed, metrics, out_dir)

        mlflow.log_metrics({"best_val_acc": best_acc, "test_loss": test_loss, "test_acc": test_acc})
        print(f"Training complete. Best val accuracy: {best_acc:.4f} "
              f"(epoch {metrics['epoch']}), test accuracy: {test_acc:.4f} (seed {seed})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministic cuDNN kernels (default on). --no-deterministic trades "
             "reproducibility for speed.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Random crop + horizontal flip on training images (default on).",
    )
    parser.add_argument(
        "--dropout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dropout in conv blocks and classifier (default on).",
    )
    parser.add_argument("--run-name", help="MLflow run name (default: derived from the flags)")
    parser.add_argument("--out-dir", help="Where to save model.pt and metrics.json")

    args = parser.parse_args()

    # Default run name from the flags: baseline / no_augment / no_dropout / no_augment_no_dropout
    disabled = [name for name, on in [("no_augment", args.augment),
                                      ("no_dropout", args.dropout)] if not on]
    run_name = args.run_name or "_".join(disabled) or "baseline"

    # The baseline goes to data/ (where serve.py looks); comparison runs go to
    # data/comparison/<run name>/
    out_dir = args.out_dir or (
        DATA_DIR if run_name == "baseline" else os.path.join(DATA_DIR, "comparison", run_name)
    )

    train(args.epochs, args.lr, args.batch_size, args.seed, args.deterministic,
          augment=args.augment, dropout=args.dropout, run_name=run_name, out_dir=out_dir)
