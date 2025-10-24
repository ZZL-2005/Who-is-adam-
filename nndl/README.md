# NNDL Project · Adam vs SGD 泛化能力研究

本项目旨在在 CIFAR-10 数据集上系统比较 Adam 与 SGD 在均方误差（MSE）与交叉熵（CE）两种损失函数下的泛化表现。仓库内包含完整的配置、数据加载、模型实现、实验脚本、日志分析与可视化工具，便于快速复现实验、记录过程并汇总结果。

## 目录结构

```
NNDL/
├── config/               # 训练超参数与优化器配置（支持继承机制）
├── datasets/             # 数据读取与增广逻辑
├── models/               # VGG/ResNet 模型定义与统一构建入口
├── output/               # 训练产物（按 run 分类、聚合结果与图表）
│   ├── runs/             # 每次实验独立目录（config、日志、checkpoint、图像）
│   ├── logs/             # 全局汇总（results.csv）
│   └── figures/          # 跨 run 绘制的额外图像
├── scripts/              # 训练、绘图、分析、评估脚本
└── train.py              # 入口脚本（调用 scripts/train.py）
```

各模块职责说明：
- `config/`: 使用 YAML 描述默认设置及优化器特定参数。`default.yaml` 给出实验通用配置，`adam_config.yaml` 与 `sgd_config.yaml` 继承默认配置并覆盖优化器参数。
- `datasets/cifar10.py`: 提供 `DataConfig` 数据类、标准化/增广流程、训练/验证划分与 DataLoader 构建，默认固定随机种子保证复现。
- `models/`: `ResNet.py` 与 `VGG.py` 适配 CIFAR-10 输出维度，`__init__.py` 暴露 `build_model` 以名称动态构建模型。
- `scripts/train.py`: 读取配置、组织多优化器/损失函数/学习率组合的实验循环，自动保存日志、checkpoint，并追加结果汇总 CSV。
- `scripts/plot_loss.py`, `scripts/analyze.py`, `scripts/evaluate.py`: 分别负责绘制损失曲线、统计不同组合的精度表现、载入 checkpoint 进行单次评估。
- `scripts/run_all.sh`: 简化批量启动训练（默认依次运行 Adam 与 SGD 配置，可附加额外命令行参数）。

## 环境准备

建议使用 Python 3.10+ 与 PyTorch 2.x。可参考如下方式创建环境并安装常见依赖：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # 根据平台替换
pip install matplotlib pandas pyyaml
```

首次运行训练脚本时会自动下载 CIFAR-10 数据集到 `./data` 目录。

## 运行训练

通过顶层入口脚本或直接调用 `scripts/train.py` 均可发起实验。如需批量运行，可使用 `scripts/run_all.sh`。

```bash
# 使用默认 ResNet18 + Adam 配置
python train.py --cfg config/adam_config.yaml

# 指定 SGD 配置，并覆盖训练轮数/输出路径
python train.py --cfg config/sgd_config.yaml --epochs 10 --output-dir ./output_sgd

# 批量运行默认的 Adam 与 SGD 组合（可附加额外参数，例如 --epochs 10）
bash scripts/run_all.sh --epochs 10
```

核心特性：
- 支持在配置文件中声明多个学习率、损失函数；脚本会自动枚举组合并生成独立 run。
- 每个 run 会产生专属目录 `output/runs/<run_name>/`，其中包含本次配置 (`config.yaml`)、原始配置拷贝、训练日志 (`metrics.csv`)、checkpoint 子目录等；运行摘要会被写入全局 `output/logs/results.csv`。
- 可通过 `--device cpu` 强制使用 CPU；若配置为 `cuda` 但不可用，脚本会自动回退到 CPU 并提示。

## 可视化与结果分析

1. 绘制损失曲线  
   ```bash
   python scripts/plot_loss.py --log output/runs/<run_name>/metrics.csv
   ```
   默认输出到 `output/runs/<run_name>/figures/loss.png`，可添加 `--show` 在屏幕上展示。

2. 汇总实验表现  
   ```bash
   python scripts/analyze.py --results output/logs/results.csv --top 5
   ```
   将按照优化器/损失分组统计平均与最佳测试精度，同时列出精度最高的若干 run。

3. 评估指定 checkpoint  
   ```bash
   python scripts/evaluate.py \
       --cfg config/adam_config.yaml \
       --checkpoint output/runs/<run_name>/checkpoints/epoch_030.pt \
       --loss cross_entropy \
       --split test
   ```

## 输出目录说明

- `output/runs/<run_name>/config.yaml`: 记录当前 run 的完整配置（含选择的优化器、损失与学习率）。
- `output/runs/<run_name>/metrics.csv`: 按 epoch 记录 train/val/test 指标。
- `output/runs/<run_name>/checkpoints/`: 保存模型权重与优化器状态（默认保留 5 份）。
- `output/runs/<run_name>/figures/`: run 专属图像输出目录。
- `output/logs/results.csv`: 汇总所有 run 的核心表现，便于后续分析。
- `output/figures/`: 如需跨 run 绘图，可手动放置在该目录。

## 结果展示（示例占位）

| 模型 | 优化器 | 损失 | 学习率 | 最佳测试精度 |
| ---- | ------ | ---- | ------ | ------------- |
| ResNet18 | Adam | CrossEntropy | 0.001 | _TBD_ |
| ResNet18 | SGD  | CrossEntropy | 0.01  | _TBD_ |

![Loss Curves Placeholder](output/runs/example_run/figures/loss.png)

> 运行训练与分析脚本后，将生成真实的日志、精度汇总与图像，可替换上述示例内容。

## 后续扩展建议

- 引入学习率调度器（如 Cosine Annealing、ReduceLROnPlateau），考察对不同优化器/损失组合的影响。
- 分析 Hessian 或 sharpness 指标，为泛化差异提供更深入解释。
- 扩展更多模型（如 WiderResNet、ViT）或其他数据集（如 CIFAR-100）以验证结论的普适性。

如需进一步定制或扩展功能，欢迎继续补充配置或脚本。祝实验顺利！
