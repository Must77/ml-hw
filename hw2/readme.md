# MLP vs CNN on Fashion-MNIST

## 快速开始
```bash
python -m venv .venv 
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch torchvision matplotlib
python main.py --epochs 10 --batch_size 128 --lr 1e-3 --out metrics.png
```

运行后会：
- 自动下载 Fashion-MNIST。
- 训练可配置的 MLP 和 CNN。
- 在同一张图中绘制训练损失与测试准确率曲线，保存到 `metrics.png`。
- 控制台输出每个模型的总参数量。

## 可调参数示例
- MLP 隐藏层结构与激活：
  ```bash
  python main.py --mlp_hidden 512 256 128 --mlp_act relu --mlp_dropout 0.2 --mlp_init xavier
  ```
- CNN 结构与正则化：
  ```bash
  python main.py --cnn_channels 32 64 128 --cnn_act leakyrelu --cnn_dropout 0.1 \
                 --cnn_global_avgpool --cnn_init kaiming
  ```
  若想关闭 BatchNorm 或改用展平 + 全连接：
  ```bash
  python main.py --cnn_no_batchnorm --cnn_use_flatten
  ```

## 完整可调参数
    ```bash
    python main.py \
        --epochs 15 \
        --batch_size 128 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --device cuda \
        --seed 42 \
        --out metrics.png \
        --mlp_hidden 512 256 128 \
        --mlp_act relu \
        --mlp_dropout 0.2 \
        --mlp_init xavier \
        --cnn_channels 32 64 \
        --cnn_act relu \
        --cnn_dropout 0.1 \
        --cnn_batchnorm \
        --cnn_global_avgpool \
        --cnn_fc_hidden 128 \
        --cnn_init kaiming
    ```

## 结果解读
- `metrics.png` 左图：训练损失曲线，右图：测试准确率曲线，图例注明模型类型与关键配置。
- 控制台打印示例：
  ```
  MLP params: 1,199,882
  CNN params:   120,778
  Epoch 001 | loss=0.60 | test_acc=0.83
  ...
  ```

## 实现要点对应需求
- MLP：任意隐藏层宽度，可选 ReLU/Tanh/LeakyReLU，支持 Dropout，权重初始化（Xavier/Kaiming/默认），`count_params` 统计总参数。
- CNN：≥2 个卷积层，Conv→激活→MaxPool 块；可选 BatchNorm / Dropout；支持全局池化或展平+全连接；同样提供参数统计。
- 训练：统一数据加载、优化器（Adam）、交叉熵损失；记录每轮训练损失与测试准确率并对比绘图。