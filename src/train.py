"""
Train the CNN on CIFAR-10 and log everything to MLflow.

CIFAR-10 is downloaded automatically on first run (~170MB).

Runs are reproducible: the seed is fixed (default 42), logged to MLflow, and
stored inside the checkpoint, and cuDNN runs in deterministic mode unless
--no-deterministic is passed.
这份文件是训练入口脚本：下载 CIFAR-10 → 增强 → 训练 CNN → 每轮评估 → 记录到 MLflow → 保存最佳模型。下面按"从上到下"逐段拆解。

Usage:
    python src/train.py
    python src/train.py --epochs 20 --lr 0.001 --batch-size 64
    python src/train.py --seed 7 --no-deterministic
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
from torchvision import datasets, transforms

# 导入 model.py里面封装好的类和三个常量
from src.model import IMAGE_SIZE, NORM_MEAN, NORM_STD, ImageClassifier

# file 是当前 train.py的路径，dirname 取出它所在的文件夹也就是 src，再拼上 .. 回到上一级，最后拼 data。
# 结果是仓库根目录下的 data 文件夹。
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# 把 data 目录的路径和文件名接在一起，得到类似 .../data/model.pt 的一段文字，存进变量 MODEL_PATH
# 也就是找 data 目录下的 model.pt 和 metrics.json这两个文件
MODEL_PATH = os.path.join(DATA_DIR, "model.pt")
METRICS_PATH = os.path.join(DATA_DIR, "metrics.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED = 42


def set_seed(seed: int, deterministic: bool = True) -> None:

    # 训练里的随机来源有四处，Python 的随机、NumPy 的随机、PyTorch 在 CPU 上的随机，以及 PyTorch 在所有 GPU 上的随机
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN kernels cost some throughput but are what makes a
    # published accuracy number reproducible on the same hardware.
    # 这两行仅对 gpu 生效
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# Data augmentation (这里注意 augmentation 是取数据的时候才做)
# prepared the training set and test set
def get_dataloaders(batch_size: int, data_dir: str):

    # Compose 本身不是数据增强，它只是一个容器，把多个预处理操作按顺序串起来
    train_transform = transforms.Compose([

        # 先在四周补 4 个像素的边到 40*40，再随机裁出 32×32，等价于随机平移 ±4 像素
        transforms.RandomCrop(IMAGE_SIZE[0], padding=4),

        # 随机水平翻转，CIFAR-10 的类别（飞机、马、卡车…）水平镜像后语义不变，但是数据量增大。
        transforms.RandomHorizontalFlip(),

        # 把图像转换为张量并进行标准化
        transforms.ToTensor(),

        # 用均值和标准差进行标准化，让输入分布更适合训练
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    # 对测试集做张量和标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    # train=True 取 5 万张训练图，train=False 取 1 万张测试图。download=True 表示本地没有就自动下载，这就是 CIFAR-10 自动下载的出处
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    # DataLoader 把数据集切成一批一批地送给模型。训练集 shuffle=True，每个 epoch 打乱顺序，测试集不打乱。num_workers=2 是用两个子进程并行读数据。
    # dataloader是一个迭代器，每次迭代的时候返回的是包含 2 的元素的序列[images, labels]

    # 27.10。2026，最开始我不理解这里，前面我们已经得到了 train_set和 test_set，为什么还要再使用 dataloader？
    # 因为train_set 是一个 Dataset，一次只能取一张图，没办法批量取图，并且可以把 batch的图转换成 tensor
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


# 每一个 epoch 结束后，都要评估一下模型在test上的表现，计算平均损失和准确率
def evaluate(model: nn.Module, loader, criterion):

    # model.eval() 切到评估模式
    # Dropout 在训练时会随机丢弃一部分神经元，评估时不丢
    # BatchNorm 用来对每一层的输入进行标准化
    model.eval()

    # 三个计数器，累计损失、答对的数量、样本总数。
    total_loss, correct, total = 0.0, 0, 0

    # torch.no_grad() 关掉梯度计算，评估时不需要反向传播，能省内存也更快 (因为 evaluate用的是训练好的模型参数，所以不需要计算梯度)。
    # 逐批处理。先把数据搬到设备上，前向得到每个类别的分数 outputs。
    # 损失乘以这一批的样本数再累加，是因为 criterion 默认返回的是平均损失，乘回去才能得到总损失，最后才能得到整体平均。
    # argmax(1) 取分数最高的类别作为预测，和真实标签比较，统计答对的个数。
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # output 指的是 model对这批images预测的类别，这里只做了 forward
            outputs = model(images) 

            # 这里.item()是为了转换数据类型，把 tensor 转换成普通数字, 为了节省内存，计算 tensor 需要占用 gpu
            # 这里的 criterion 返回的是每一张图的平均loss，而不是总的 loss
            total_loss += criterion(outputs, labels).item() * len(labels)

            # 这里计算的是这一个 batch 预测对了几张图
            # outputs.argmax(1)指的是取每一行中最大的值的索引，然后判断是否和 label 一致，true 是为 1
            correct += (outputs.argmax(1) == labels).sum().item()
            
            # 计算的是总图片数
            total += len(labels)
        
    # 返回平均损失和准确率
    return total_loss / total, correct / total



# save_checkpoint 的作用是把训练出来的模型存到硬盘上，并且把它是怎么训练出来的、成绩如何，一起存进去
def save_checkpoint(model: nn.Module, params: dict, seed: int, metrics: dict) -> None:

    # MODEL_PATH 是 data/model.pt
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 不只存权重，还把超参数、种子、指标和类别名一起打包存进同一个文件。
    # 这样以后拿到一个 model.pt，就知道它是怎么训练出来的、成绩如何。
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": params,
            "seed": seed,
            "metrics": metrics,
            "classes": ImageClassifier.CLASSES, # 访问类属性
        },
        MODEL_PATH,
    )

    # 另外把参数、种子、指标写成人能直接读的 json 文件，indent=2 是缩进两格方便阅读
    with open(METRICS_PATH, "w") as f:
        json.dump({"params": params, "seed": seed, "metrics": metrics}, f, indent=2)



# 先设种子，再告诉 MLflow 这次实验叫 image-classifier，所有运行会被归到这个名字下面
def train(
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int = DEFAULT_SEED,
    deterministic: bool = True,
):
    set_seed(seed, deterministic)

    data_dir = DATA_DIR
    mlflow.set_experiment("image-classifier")

# 把这次运行的配置收集成一个字典，后面既要上传给 MLflow, 也要写进检查点
    params = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "deterministic": deterministic,
        "device": DEVICE,
    }

    # 每一个 mlflow的超参数都会被记录下来
    with mlflow.start_run():
        mlflow.log_params(params)

        train_loader, test_loader = get_dataloaders(batch_size, data_dir)
        
        # # 调用 imageclassifier类，然后放到设备上用 gpu/cpu跑
        model = ImageClassifier().to(DEVICE) 

        # 注意这里 criterion 返回的是每一张图的平均loss，而不是总的 loss
        criterion = nn.CrossEntropyLoss() 

        # optimizer负责 gradient 更新权重。model.parameters() 告诉它要更新哪些权重，weight_decay 是防过拟合的正则项。
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            total_acc = 0
            total = 0 # 指的是已经处理过的图片张数

            # 每一个 batch 结束后更新参数，第二个 batch 基于第一个 batch 更新参数
            # 每一个 batch 计算的时候都要清空上一步的梯度
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                # zero_grad 清空上一步留下的梯度，否则梯度会累加
                # 前向计算并算出损失
                # backward 反向传播，算出每个参数的梯度
                # step 用梯度更新参数
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                # backward 反向传播，算出每个参数的梯度
                loss.backward()

                # step 用梯度更新参数
                optimizer.step()

                # 计算 training loss 和 acc
                total_loss += loss.item() * len(labels)
                total_acc += (outputs.argmax(1) == labels).sum().item()
                total += len(labels)
            
            train_loss = total_loss / total
            train_acc = total_acc / total

            scheduler.step() # 每个 epoch 结束后更新一次学习率


            val_loss, val_acc = evaluate(model, test_loader, criterion)
            # 记录每一个 epoch 的 validation loss 和 accuracy，step=epoch 表示这是第几个 epoch
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

        
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    model,
                    params,
                    seed,
                    {"val_loss": val_loss, "val_acc": val_acc, "best_val_acc": best_acc,
                     "epoch": epoch},
                )

        mlflow.log_metric("best_val_acc", best_acc)
        print(f"Training complete. Best accuracy: {best_acc:.4f} (seed {seed})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
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

    # 解析命令行，把参数传给 train 函数开始训练
    args = parser.parse_args()
    train(args.epochs, args.lr, args.batch_size, args.seed, args.deterministic)
