"""
CNN architecture for CIFAR-10 image classification.

Input: 32x32 RGB images
Output: 10 class probabilities

Architecture:
    Block 1: Conv(32) -> BN -> ReLU -> Conv(32) -> BN -> ReLU -> MaxPool -> Dropout
    Block 2: Conv(64) -> BN -> ReLU -> Conv(64) -> BN -> ReLU -> MaxPool -> Dropout
    Block 3: Conv(128) -> BN -> ReLU -> Conv(128) -> BN -> ReLU -> MaxPool -> Dropout
    Classifier: FC(512) -> ReLU -> Dropout -> FC(10)
"""

import torch
from torch import nn


# image size
IMAGE_SIZE = (32, 32)

# 三个 channels RGB的均值
NORM_MEAN = (0.4914, 0.4822, 0.4465)

# 三个通道的标准差
NORM_STD = (0.2470, 0.2435, 0.2616)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # 让每一层的输入分布更稳定，训练更快
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 缩小特征图，减少后面的计算量
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImageClassifier(nn.Module):

    # CIFAR10 数据集中共 10 个类别，0-9 分别表示这 10 个类别
    CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        # 这部分就是所谓的 fully connected
        self.classifier = nn.Sequential(
            # 把 feature map拉成一个向量
            nn.Flatten(),

            # 降维
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes), # 模型的最终结果是未经过 normalize 的数据，而不是 proba
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
