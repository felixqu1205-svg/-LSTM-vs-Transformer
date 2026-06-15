# -LSTM-vs-Transformer
已确定选题方向为「BiLSTM 与 Transformer Encoder 双路对比实验」，任务为中文影评情感分析，在 RTX 4060 上对比训练效率、收敛速度与分类性能，并完成 AMP 混合精度 Profile 分析。
# 中文影评情感分析 · LSTM vs Transformer

## 项目结构

```
config.py          配置
data.py            数据加载与生成
models.py          Bi-LSTM / Transformer
train.py           训练引擎
run_train.py       一键训练（PyCharm 运行）
run_demo.py        答辩演示（PyCharm 运行）
data/chinese_sentiment.json   中文数据集（10000+1000）
outputs/           训练后自动生成（模型/图表/记录）
```

## 运行

1. `pip install -r requirements.txt`
2. PyCharm 运行 **`run_train.py`**（完整训练）
3. PyCharm 运行 **`run_demo.py`**（中文演示）

## 数据

- 训练 10000 条 + 测试 1000 条


## TensorBoard

```bash
tensorboard --logdir D:\pythonProject\.tb_logs\sentiment_zh
```
