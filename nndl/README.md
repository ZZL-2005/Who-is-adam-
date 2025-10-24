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


## 环境准备


## 运行训练

通过顶层入口脚本或直接调用 `scripts/train.py` 均可发起实验。如需批量运行，可使用 `scripts/run_all.sh`。

```bash
# 使用默认 ResNet18 + Adam 配置(KEy)
python -m scripts.train --cfg config/adam_config.yaml --lr 0.1 --output-dir ./output/adam

# 指定 SGD 配置，并覆盖训练轮数/输出路径
python train.py --cfg config/sgd_config.yaml --epochs 10 --output-dir ./output_sgd

```


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


## 结果展示（示例占位）


## 后续扩展建议

- 引入学习率调度器（如 Cosine Annealing、ReduceLROnPlateau），考察对不同优化器/损失组合的影响。
- 分析 Hessian 或 sharpness 指标，为泛化差异提供更深入解释。
- 扩展更多模型（如 WiderResNet、ViT）或其他数据集（如 CIFAR-100）以验证结论的普适性。

如需进一步定制或扩展功能，欢迎继续补充配置或脚本。祝实验顺利！
