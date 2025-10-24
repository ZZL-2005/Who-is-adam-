非常清晰！你的实验目标是通过 **CIFAR-10 数据集** 比较 **Adam** 和 **SGD** 在 **MSE** 与 **CE** 损失函数下的 **泛化能力**，同时需要详细记录每个训练 epoch 的 **loss**，并保存所有的 **checkpoint**。为了帮助 **Codex** 高效生成代码，我们可以明确每个步骤并且设置好框架。以下是一个详细的 **TODO.md** 文件，它将为你在 **Codex** 生成代码时提供清晰的指引。

---

## ✅ **TODO.md - NNDL Project**

```markdown
# NNDL Project - Adam vs SGD 泛化能力研究

## 🎯 研究目标
本项目旨在探究 **Adam** 和 **SGD** 在不同损失函数（**MSE** 和 **CE**）下的泛化能力差异。使用 **CIFAR-10** 数据集，并配置不同的学习率（`0.001`, `0.005`, `0.01`, `0.1`）进行训练。实验中需要记录每个 epoch 的训练过程，存储所有中间 **loss** 和 **checkpoint**。

## 🧩 项目结构
```

NNDL/
├── config/            # 📦 配置文件（超参数配置）
│   ├── default.yaml   # 基础配置（学习率、epochs）
│   ├── adam_config.yaml  # Adam 配置
│   └── sgd_config.yaml  # SGD 配置
│
├── datasets/          # 📂 数据集加载与处理
│   └── cifar10.py     # CIFAR-10 数据加载脚本
│
├── models/            # 🧩 模型实现
│   ├── VGG.py         # VGG 模型
│   ├── ResNet.py      # ResNet 模型
│   └── **init**.py    # 模型构建入口
│
├── scripts/           # 🧪 训练与实验脚本
│   ├── train.py       # 训练脚本（主脚本）
│   ├── plot_loss.py   # 绘制训练过程中 loss 曲线
│   ├── analyze.py     # 分析与结果汇总脚本
│   └── evaluate.py    # 测试与评估脚本
│
├── output/            # 📊 保存日志、模型
│   ├── checkpoints/   # 模型检查点保存目录
│   ├── logs/          # 训练过程日志
│   └── figures/       # 绘制的图形（loss曲线、acc）
│
├── README.md          # 项目介绍与运行说明
└── train.py           # 入口脚本（用于启动训练）

````

---

## ✅ **1. 配置文件管理 (config)**

### 1.1 **`config/default.yaml`**
- 配置所有基础的超参数（如：`epochs`, `batch_size`, `seed`, `device`）

```yaml
# 默认配置
seed: 42
epochs: 30
batch_size: 128
device: "cuda"  # "cpu" / "cuda"

learning_rates: [0.001, 0.005, 0.01, 0.1]  # 四个学习率档位
````

### 1.2 **`config/adam_config.yaml`**

* 配置 Adam 优化器的超参数（如 `lr`, `betas`, `weight_decay`）

```yaml
optimizer: 
  type: "adam"
  lr: 0.001
  betas: [0.9, 0.999]
  weight_decay: 0.0
```

### 1.3 **`config/sgd_config.yaml`**

* 配置 SGD 优化器的超参数（如 `lr`, `momentum`, `weight_decay`）

```yaml
optimizer:
  type: "sgd"
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0
```

---

## ✅ **2. 数据集加载 (datasets)**

### 2.1 **`datasets/cifar10.py`**

* 负责加载 **CIFAR-10** 数据集并进行预处理
* 使用 `torchvision` 提供的标准接口加载数据集并进行标准化。

```python
import torch
from torchvision import datasets, transforms

def get_dataloader(batch_size=128, train=True, seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return dataloader
```

---

## ✅ **3. 模型定义 (models)**

### 3.1 **`models/VGG.py`**

* 定义 **VGG** 网络，并支持 CIFAR-10 版输出层。

```python
import torch.nn as nn
import torchvision.models as models

def VGG11():
    model = models.vgg11_bn(weights=None)
    model.classifier[6] = nn.Linear(4096, 10)  # CIFAR-10 分类
    return model
```

### 3.2 **`models/ResNet.py`**

* 定义 **ResNet18** 网络，调整输出层以适应 CIFAR-10。

```python
import torch.nn as nn
import torchvision.models as models

def ResNet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 分类
    return model
```

### 3.3 **`models/__init__.py`**

* 提供一个统一入口，用于加载不同的模型。

```python
from .VGG import VGG11
from .ResNet import ResNet18

def build_model(model_name="ResNet18"):
    if model_name == "VGG11":
        return VGG11()
    elif model_name == "ResNet18":
        return ResNet18()
    else:
        raise ValueError("Model not found!")
```

---

## ✅ **4. 训练脚本 (scripts/train.py)**

### 4.1 **训练流程**

* 加载配置文件，选择模型、优化器、损失函数。
* 对于每个学习率档次运行训练过程，记录训练日志和模型。
* 每个 epoch 结束时保存 **checkpoint**。

```python
import torch
import torch.optim as optim
import torch.nn as nn
from models import build_model
from datasets.cifar10 import get_dataloader
import yaml
import os

# 加载配置
with open('config/adam_config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = torch.device(config['device'])

# 模型、优化器
model = build_model("ResNet18").to(device)
optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['lr'], betas=config['optimizer']['betas'])

# 数据加载
train_loader = get_dataloader(batch_size=128)
test_loader = get_dataloader(batch_size=128, train=False)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(1, config['epochs'] + 1):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch}, Loss: {running_loss / len(train_loader)}')

    # 保存模型 checkpoint
    torch.save(model.state_dict(), f'output/checkpoints/model_epoch_{epoch}.pth')
```

---

## ✅ **5. 可视化 (scripts/plot_loss.py)**

### 5.1 **实时绘制训练损失**

* 使用 `matplotlib` 进行实时损失绘制。

```python
import matplotlib.pyplot as plt

def plot_loss(train_losses, test_losses):
    plt.ion()
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.pause(0.1)
```

---

## ✅ **6. 结果分析与评估 (scripts/analyze.py)**

### 6.1 **分析不同优化器与损失下的表现**

* 统计每个学习率档次的 **test_acc**, **Hessian**, **sharpness**，并绘制对比图。

```python
import pandas as pd
import matplotlib.pyplot as plt

def analyze_results():
    df = pd.read_csv('output/logs/results.csv')
    df.groupby('learning_rate')['test_acc'].mean().plot()
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Test Accuracy')
    plt.title('Adam vs SGD Generalization')
    plt.show()
```

---

## ✅ **7. Checkpoints & 结果存储**

* 每次训练后保存 **checkpoint**：

  * 训练过程的日志、每个 epoch 的模型。
  * 存储 `output/checkpoints` 和 `output/logs/`。

---

## ✅ **8. 未来扩展**

* [ ] **引入学习率调度器**（如 `ReduceLROnPlateau`）。
* [ ] **Hessian 分析**：对比不同模型、损失下的 **Hessian 谱**。
* [ ] **Sharpness Proxy**：量化模型平坦度。
* [ ] **多任务训练**：通过多模型（ResNet / VGG）和不同任务的对比进一步验证。

---

## ✅ **9. 运行指南**

1. 配置文件

   ```bash
   python train.py --cfg config/adam_config.yaml
   ```

2. 运行训练后，可视化损失：

   ```bash
   python scripts/plot_loss.py
   ```

3. 结果分析：

   ```bash
   python scripts/analyze.py
   ```

---

## ✅ **结论**

这个 `TODO.md` 结构清晰，任务明确，可以用来引导 **Codex** 撰写代码。每个子任务可以单独进行实现，逐步构建完整的实验流程。如果有任何进一步需求，欢迎随时调整或补充！
