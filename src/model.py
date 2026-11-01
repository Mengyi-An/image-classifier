"""
CNN architecture for CIFAR-10 image classification.

Input: 32x32 RGB images
Output: 10 class scores (logits)

Architecture:
    Block 1: Conv(32) -> BN -> ReLU -> Conv(32) -> BN -> ReLU -> MaxPool -> Dropout
    Block 2: Conv(64) -> BN -> ReLU -> Conv(64) -> BN -> ReLU -> MaxPool -> Dropout
    Block 3: Conv(128) -> BN -> ReLU -> Conv(128) -> BN -> ReLU -> MaxPool -> Dropout
    Classifier: FC(512) -> ReLU -> Dropout -> FC(10)
"""

import torch
from torch import nn

IMAGE_SIZE = (32, 32)

# Per-channel (R, G, B) mean and standard deviation of the CIFAR-10 training set
NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2470, 0.2435, 0.2616)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # stabilizes layer inputs and speeds up training
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # halves the feature map size, reducing later computation
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImageClassifier(nn.Module):

    # The 10 CIFAR-10 classes; label i corresponds to CLASSES[i]
    CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

    # dropout is the rate in the classifier, conv_dropout the rate at the end of each conv
    # block. Setting both to 0 disables dropout (used in the comparison runs). Dropout layers
    # have no weights, so the saved state_dict is the same for any rate, and serve.py can
    # load any checkpoint with the default values.
    def __init__(self, num_classes: int = 10, dropout: float = 0.5, conv_dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, conv_dropout),
            ConvBlock(32, 64, conv_dropout),
            ConvBlock(64, 128, conv_dropout),
        )

        # Fully connected classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 128 feature maps of 4x4 -> one vector of 2048 values
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),  # outputs raw logits, not probabilities
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
