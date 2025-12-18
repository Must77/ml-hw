import argparse
import os
from typing import List, Callable, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_activation(name: str) -> Callable[[], nn.Module]:
    """
    根据字符串返回对应的激活函数类（构造器）。
    这样在构建网络时可以按需替换激活函数。
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "leakyrelu":
        return lambda: nn.LeakyReLU(negative_slope=0.1)
    raise ValueError(f"Unsupported activation: {name}")


def init_weights(module: nn.Module, scheme: str):
    """
    按指定方案初始化线性层或卷积层的权重。
    - 默认（default）不做处理，使用 PyTorch 内建初始化。
    - Xavier/Kaiming 适合深层网络的稳定训练。
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if scheme == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif scheme == "kaiming":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        # 对偏置统一置零
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLP(nn.Module):
    """
    可配置多层感知机：
    - 支持任意隐藏层宽度列表
    - 可选激活函数、Dropout
    - 可选初始化方案
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        activation: str = "relu",
        dropout: float = 0.0,
        init_scheme: str = "default",
    ):
        super().__init__()
        act = get_activation(activation)
        layers = []
        prev = input_dim
        # 堆叠隐藏层
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        # 输出层（不加激活，交给 CrossEntropyLoss 内置 softmax 处理）
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

        # 递归对可初始化的层进行权重初始化
        self.apply(lambda m: init_weights(m, init_scheme))

    def forward(self, x):
        # 将 2D 图像展平为向量： [B, 1, 28, 28] -> [B, 784]
        x = torch.flatten(x, 1)
        return self.net(x)

    def count_params(self) -> int:
        # 统计模型总参数量
        return sum(p.numel() for p in self.parameters())


class SimpleCNN(nn.Module):
    """
    基础卷积神经网络：
    - 多个 Conv2d -> (可选 BN) -> 激活 -> MaxPool 组成的卷积块
    - 支持 Dropout2d 正则
    - 支持全局平均池化或展平后接全连接
    - 同样提供参数统计
    """
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64],
        kernel_size: int = 3,
        activation: str = "relu",
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        use_global_avgpool: bool = True,
        fc_hidden: int = 128,
        num_classes: int = 10,
        init_scheme: str = "default",
    ):
        super().__init__()
        act = get_activation(activation)
        convs = []
        prev_c = in_channels
        # 构造卷积块序列
        for c in conv_channels:
            convs.append(nn.Conv2d(prev_c, c, kernel_size=kernel_size, padding=kernel_size // 2))
            if use_batchnorm:
                convs.append(nn.BatchNorm2d(c))
            convs.append(act())
            convs.append(nn.MaxPool2d(kernel_size=2))  # 空间尺寸减半
            if dropout > 0:
                convs.append(nn.Dropout2d(dropout))
            prev_c = c
        self.features = nn.Sequential(*convs)

        # 分类头：全局平均池化 or 展平 + 全连接
        self.use_global_avgpool = use_global_avgpool
        if use_global_avgpool:
            # 自适应池化到 1x1，并使用线性层映射到类别
            self.classifier = nn.Linear(prev_c, num_classes)
        else:
            # 对 Fashion-MNIST：28x28，经过两次 2x2 池化 -> 7x7
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(prev_c * 7 * 7, fc_hidden),
                act(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(fc_hidden, num_classes),
            )

        # 应用权重初始化
        self.apply(lambda m: init_weights(m, init_scheme))

    def forward(self, x):
        x = self.features(x)
        if self.use_global_avgpool:
            # 自适应平均池化到 1x1，然后展平
            x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
            x = self.classifier(x)
        else:
            x = self.classifier(x)
        return x

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def get_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    下载/加载 Fashion-MNIST，并构造训练与测试 DataLoader。
    - ToTensor 会把图像转为 [0,1] 张量并通道前置。
    - num_workers=2 在多数环境下较为稳妥。
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    在测试集上评估分类准确率。
    使用 no_grad 关闭梯度，节省显存和加速推理。
    """
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Tuple[list, list]:
    """
    训练单个模型并记录：
    - 每个 epoch 的训练损失（平均交叉熵）
    - 对应测试集准确率
    返回 (train_losses, test_accs) 供可视化。
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)  # 按样本数累计总损失

        epoch_loss = running_loss / len(train_loader.dataset)  # 归一化到平均损失
        train_losses.append(epoch_loss)

        # 每轮结束在测试集评估准确率，便于观察收敛和泛化
        test_acc = evaluate(model, test_loader, device)
        test_accs.append(test_acc)
        print(f"Epoch {epoch+1:03d} | loss={epoch_loss:.4f} | test_acc={test_acc:.4f}")
    return train_losses, test_accs


def plot_curves(
    mlp_hist: Tuple[list, list],
    cnn_hist: Tuple[list, list],
    mlp_label: str,
    cnn_label: str,
    out_path: str,
):
    """
    将 MLP 与 CNN 的训练损失、测试准确率绘制在同一张图中。
    - 左图：训练损失对比
    - 右图：测试准确率对比
    图例包含模型类型与关键配置，便于阅读。
    """
    mlp_losses, mlp_accs = mlp_hist
    cnn_losses, cnn_accs = cnn_hist
    epochs = range(1, len(mlp_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, mlp_losses, label=f"MLP ({mlp_label})")
    axes[0].plot(epochs, cnn_losses, label=f"CNN ({cnn_label})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, mlp_accs, label=f"MLP ({mlp_label})")
    axes[1].plot(epochs, cnn_accs, label=f"CNN ({cnn_label})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("Test Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved curves to {out_path}")


def parse_args():
    """
    命令行参数：
    - 训练相关：epochs, batch_size, lr, weight_decay, device, seed, out
    - MLP 配置：隐藏层、激活、Dropout、初始化
    - CNN 配置：通道数、激活、Dropout、BatchNorm、池化方式、初始化
    """
    parser = argparse.ArgumentParser(description="MLP vs CNN on Fashion-MNIST")
    # shared
    parser.add_argument("--epochs", 
                        type=int, default=10, 
                        help="训练的总轮数")

    parser.add_argument("--batch_size", 
                        type=int, default=128, 
                        help="每个 batch 的样本数")

    parser.add_argument("--lr", 
                        type=float, default=1e-3, 
                        help="Adam 优化器的学习率")

    parser.add_argument("--weight_decay", 
                        type=float, default=1e-4, 
                        help="L2 权重衰减系数（正则化）")

    parser.add_argument("--device", 
                        type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="训练设备：cuda 或 cpu")

    parser.add_argument("--seed", 
                        type=int, default=42, 
                        help="随机种子，保证可复现")

    parser.add_argument("--out", 
                        type=str, default="metrics.png", 
                        help="训练曲线输出图片路径")


    # MLP
    parser.add_argument("--mlp_hidden", 
                        type=int, nargs="+", default=[512, 256, 128],
                        help="MLP 隐藏层宽度列表，如 512 256 128")

    parser.add_argument("--mlp_act", 
                        type=str, default="relu", choices=["relu", "tanh", "leakyrelu"],
                        help="MLP 激活函数类型")

    parser.add_argument("--mlp_dropout", 
                        type=float, default=0.2, 
                        help="MLP 隐藏层后的 Dropout 概率")

    parser.add_argument("--mlp_init", 
                        type=str, default="xavier", choices=["xavier", "kaiming", "default"],
                        help="MLP 权重初始化方式")


    # CNN
    parser.add_argument("--cnn_channels", 
                        type=int, nargs="+", default=[32, 64],
                        help="CNN 每个卷积层的输出通道列表")

    parser.add_argument("--cnn_act", 
                        type=str, default="relu", choices=["relu", "tanh", "leakyrelu"],
                        help="CNN 激活函数类型")

    parser.add_argument("--cnn_dropout", 
                        type=float, default=0.1, 
                        help="CNN 中 Dropout/Dropout2d 概率")

    parser.add_argument("--cnn_batchnorm", 
                        action="store_true", default=True,
                        help="是否在卷积层后使用 BatchNorm（默认开启）")

    parser.add_argument("--cnn_no_batchnorm", dest="cnn_batchnorm", 
                        action="store_false",
                        help="关闭 BatchNorm 时使用该标志")

    parser.add_argument("--cnn_global_avgpool", 
                        action="store_true", default=True,
                        help="是否使用全局平均池化接线性层（默认是），否则使用展平+全连接")

    parser.add_argument("--cnn_use_flatten", dest="cnn_global_avgpool", 
                        action="store_false",
                        help="改用展平+全连接分类头时使用该标志")

    parser.add_argument("--cnn_fc_hidden", 
                        type=int, default=128,
                        help="展平路径下全连接隐藏层宽度（仅在未使用全局池化时有效）")

    parser.add_argument("--cnn_init", 
                        type=str, default="kaiming", choices=["xavier", "kaiming", "default"],
                        help="CNN 权重初始化方式")


    return parser.parse_args()


def main():
    """
    主流程：
    1) 解析参数 & 设置随机种子
    2) 构造数据加载器
    3) 初始化 MLP 和 CNN
    4) 分别训练并记录损失/准确率
    5) 绘图对比并保存
    """
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    train_loader, test_loader = get_loaders(args.batch_size)

    # 实例化 MLP
    mlp = MLP(
        input_dim=28 * 28,  # Fashion-MNIST 单通道 28x28
        hidden_dims=args.mlp_hidden,
        num_classes=10,
        activation=args.mlp_act,
        dropout=args.mlp_dropout,
        init_scheme=args.mlp_init,
    )
    # 实例化 CNN
    cnn = SimpleCNN(
        in_channels=1,
        conv_channels=args.cnn_channels,
        activation=args.cnn_act,
        use_batchnorm=args.cnn_batchnorm,
        dropout=args.cnn_dropout,
        use_global_avgpool=args.cnn_global_avgpool,
        fc_hidden=args.cnn_fc_hidden,
        num_classes=10,
        init_scheme=args.cnn_init,
    )

    # 打印参数量，便于比较模型大小/容量
    print(f"MLP params: {mlp.count_params():,}")
    print(f"CNN params: {cnn.count_params():,}")

    # 训练并记录历史
    mlp_hist = train_one_model(mlp, train_loader, test_loader, device, args.epochs, args.lr, args.weight_decay)
    cnn_hist = train_one_model(cnn, train_loader, test_loader, device, args.epochs, args.lr, args.weight_decay)

    # 在图例中标注关键信息，便于阅读
    mlp_label = f"{args.mlp_act}, dropout={args.mlp_dropout}, init={args.mlp_init}"
    cnn_label = f"{args.cnn_act}, dropout={args.cnn_dropout}, bn={args.cnn_batchnorm}, init={args.cnn_init}"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    plot_curves(mlp_hist, cnn_hist, mlp_label, cnn_label, args.out)


if __name__ == "__main__":
    main()