"""
Train the CNN on CIFAR-10 and log everything to MLflow.

CIFAR-10 is downloaded automatically on first run (~170MB).

Runs are reproducible: the seed is fixed (default 42), logged to MLflow, and
stored inside the checkpoint, and cuDNN runs in deterministic mode unless
--no-deterministic is passed.
这份文件是训练入口脚本：下载 CIFAR-10 → 增强 → 训练 CNN → 每轮评估 → 记录到 MLflow → 保存最佳模型。下面按"从上到下"逐段拆解。

The 50k training images are split into 45k train / 5k validation (fixed split).
The best epoch is picked on validation; the test set is used exactly once at the end.

Usage:
    python -m src.train
    python -m src.train --epochs 20 --lr 0.001 --batch-size 64
    python -m src.train --seed 7 --no-deterministic

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

# 从 5 万张训练图里切出 5000 张当验证集。切分用单独的固定种子，
# 这样换训练种子、做消融实验时，所有 run 用的都是同一份验证集，结果才能直接比较
VAL_SIZE = 5000
SPLIT_SEED = 0

# Compose 本身不是数据增强，它只是一个容器，把多个预处理操作按顺序串起来
TRAIN_TRANSFORM = transforms.Compose([

    # 先在四周补 4 个像素的边到 40*40，再随机裁出 32×32，等价于随机平移 ±4 像素
    transforms.RandomCrop(IMAGE_SIZE[0], padding=4),

    # 随机水平翻转，CIFAR-10 的类别（飞机、马、卡车…）水平镜像后语义不变，但是数据量增大。
    transforms.RandomHorizontalFlip(),

    # 把图像转换为张量并进行标准化
    transforms.ToTensor(),

    # 用均值和标准差进行标准化，让输入分布更适合训练
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# 这里针对 validation set和 test set，只做张量和标准化，不做增强
EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def set_seed(seed: int, deterministic: bool = True) -> None:

    # 训练里的随机来源有四处，Python 的随机、NumPy 的随机、PyTorch 在 CPU 上的随机，以及 PyTorch 在所有 GPU 上的随机
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 这两行仅对 gpu 生效
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# Data augmentation (这里注意 augmentation 是取数据的时候才做)
# prepared the training, validation and test sets
# augment=False 时训练集也不做增强（消融实验用）
def get_dataloaders(batch_size: int, data_dir: str, augment: bool = True):
    train_transform = TRAIN_TRANSFORM if augment else EVAL_TRANSFORM

    # train=True 取 5 万张训练图，train=False 取 1 万张测试图。download=True 表示本地没有就自动下载，这就是 CIFAR-10 自动下载的出处
    # 同一份 5 万张训练图读两遍：一份带增强用来训练，一份不带增强用来评估
    train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    clean_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=EVAL_TRANSFORM)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=EVAL_TRANSFORM)

    # 用固定种子打乱下标，前 5000 个当验证集，剩下 45000 个当训练集
    generator = torch.Generator().manual_seed(SPLIT_SEED) # 新建一个专属的随机数生成器，可以理解为一个"骰子"，将数据集打乱

    # random permutation（随机排列），把 0, 1, ..., n-1 这 n 个数随机打乱顺序。每个数恰好出现一次，不重复也不遗漏。
    # .tolist()指的是把 tensor 转换成 list
    indices = torch.randperm(len(train_full), generator=generator).tolist() 
    val_idx, train_idx = indices[:VAL_SIZE], indices[VAL_SIZE:]

    train_set = Subset(train_full, train_idx)
    val_set = Subset(clean_full, val_idx)

    # 训练时的 train_acc 是在增强 + dropout 打开的状态下算的，会偏低，不能直接和 val_acc 比。
    # 这里就是在做 augmentation 之前，把 training set 复制一份作为 clean set，然后从 clean set 里取数据，train_val_set大小和 val_set大小一致。
    train_eval_set = Subset(clean_full, train_idx[:VAL_SIZE])

    # DataLoader 把数据集切成一批一批地送给模型。训练集 shuffle=True，每个 epoch 打乱顺序，测试集不打乱。num_workers=2 是用两个子进程并行读数据。
    # dataloader是一个迭代器，每次迭代的时候返回的是包含 2 的元素的序列[images, labels]

    # 27.10。2026，最开始我不理解这里，前面我们已经得到了 train_set和 test_set，为什么还要再使用 dataloader？
    # 因为train_set 是一个 Dataset，一次只能取一张图，没办法批量取图，并且可以把 batch的图转换成 tensor
    def make_loader(dataset, shuffle: bool = False):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return (

        # 这里 training set的 shuffle=true是因为数据集通常是按照类别排序的，比如前 5000 张都是飞机，接下来 5000 张都是汽车
        make_loader(train_set, shuffle=True),
        make_loader(train_eval_set),
        make_loader(val_set),
        make_loader(test_set),
    )


# 每一个 epoch 结束后，都要评估一下模型在验证集上的表现，计算平均损失和准确率
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
            # 这里的 criterion 返回的是是 "这一个 batch 里所有图的平均 loss
            # len(labels)就是这一个 batch中的图片的数量
            # 最开始我想的是为什么这里不乘以 batch_size，但是通常最后一个 batch 通常不满
            total_loss += criterion(outputs, labels).item() * len(labels)

            # 这里计算的是这一个 batch 预测对了几张图
            # outputs.argmax(1)指的是取每一行中最大的值的索引，然后判断是否和 label 一致，true 是为 1
            correct += (outputs.argmax(1) == labels).sum().item()
            
            # 计算的是总图片数
            total += len(labels)
        
    # 返回平均损失和准确率
    return total_loss / total, correct / total



# save_checkpoint 的作用是把训练出来的模型存到硬盘上，并且把它是怎么训练出来的、成绩如何，一起存进去
# out_dir 默认是 data/，消融实验存到 data/ablation/<run 名>/，不会覆盖 serve.py 用的 data/model.pt
def save_checkpoint(
    model: nn.Module, params: dict, seed: int, metrics: dict, out_dir: str = DATA_DIR
) -> None:
    os.makedirs(out_dir, exist_ok=True)

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
        os.path.join(out_dir, "model.pt"),
    )

    # 另外把参数、种子、指标写成人能直接读的 json 文件，indent=2 是缩进两格方便阅读
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"params": params, "seed": seed, "metrics": metrics}, f, indent=2)



# 先设种子，再告诉 MLflow 这次实验叫 image-classifier，所有运行会被归到这个名字下面
def train(
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int = DEFAULT_SEED,
    deterministic: bool = True,
    *, # *的意思是：它后面的参数调用时必须写参数名，比如 augment=False
    augment: bool = True, # 是否做数据增强
    dropout: bool = True, # 是否 dropput
    run_name: str = "baseline", # 每一次实验名称记录下来
    out_dir: str = DATA_DIR,
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
        "augment": augment,
        "dropout": dropout,
        "val_size": VAL_SIZE,
    }

    # 每一个 mlflow的超参数都会被记录下来。run_name 用来在 MLflow 里区分消融实验的各组
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        train_loader, train_eval_loader, val_loader, test_loader = get_dataloaders(
            batch_size, data_dir, augment
        )

        # dropout = true / false
        if dropout:
            model = ImageClassifier().to(DEVICE)
        else:
            model = ImageClassifier(dropout=0.0, conv_dropout=0.0).to(DEVICE)

        # 注意这里 criterion 返回的是当前 batch的平均loss，而不是总的 loss
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


            train_eval_loss, train_eval_acc = evaluate(model, train_eval_loader, criterion)
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            # 记录每一个 epoch 的指标，step=epoch 表示这是第几个 epoch
            # train_eval 和 val 之间的差距（过拟合程度）在 plot.py 里画图时直接看两条曲线
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_eval_loss": train_eval_loss,
                    "train_eval_acc": train_eval_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch
            )

            print(f"Epoch {epoch}/{epochs}  "
                  f"train_loss={train_loss:.4f}  train_eval_acc={train_eval_acc:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")


            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    model,
                    params,
                    seed,
                    {"val_loss": val_loss, "val_acc": val_acc, "best_val_acc": best_acc,
                     "train_eval_acc": train_eval_acc, "epoch": epoch},
                    out_dir,
                )

        # 训练结束后，载入val_set上最好的那一轮，在test上只评估这一次，这个数字才是无偏的最终成绩
        # 载入最好的模型 → 在测试集上评估 → 把测试成绩存回文件 → 记录并打印结果。
        checkpoint = torch.load(
            os.path.join(out_dir, "model.pt"), map_location=DEVICE, weights_only=True
        )

        # 把最好的权重等参数放回模型，.load_state_dict()函数指的是把字典里的参数放回模型
        model.load_state_dict(checkpoint["state_dict"])
        test_loss, test_acc = evaluate(model, test_loader, criterion) # 然后用最好的模型跑 test set

        metrics = {**checkpoint["metrics"], "test_loss": test_loss, "test_acc": test_acc}
        save_checkpoint(model, params, seed, metrics, out_dir)

        mlflow.log_metrics({"best_val_acc": best_acc, "test_loss": test_loss, "test_acc": test_acc})
        print(f"Training complete. Best val accuracy: {best_acc:.4f} "
              f"(epoch {metrics['epoch']}), test accuracy: {test_acc:.4f} (seed {seed})")


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

    # 解析命令行，把参数传给 train 函数开始训练
    args = parser.parse_args()

    # 没指定 run 名时按开关自动起名：baseline / no_augment / no_dropout / no_augment_no_dropout
    ablations = [name for name, on in [("no_augment", args.augment),
                                       ("no_dropout", args.dropout)] if not on]
    run_name = args.run_name or "_".join(ablations) or "baseline"

    # baseline 存到 data/（serve.py 读的就是这里），其他组存到 data/ablation/<run 名>/
    out_dir = args.out_dir or (
        DATA_DIR if run_name == "baseline" else os.path.join(DATA_DIR, "ablation", run_name)
    )

    train(args.epochs, args.lr, args.batch_size, args.seed, args.deterministic,
          augment=args.augment, dropout=args.dropout, run_name=run_name, out_dir=out_dir)
