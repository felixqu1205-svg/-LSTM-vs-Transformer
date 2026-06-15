"""PyCharm 一键训练：直接运行本文件，每次执行完整 7 步流程。"""

import csv
import json
import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib.pyplot as plt
import torch

from config import (
    CKPT_DIR, FIG_DIR, LSTM_EPOCHS, LSTM_EXPS, PROFILE_EPOCHS, RESULTS_DIR, SEED, TRANS_EPOCHS,
)
from data import load_data
from models import BiLSTMClassifier, TransformerClassifier
from train import run_training, test_model

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def _bar(title, names, values, path, ylabel=""):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, values, color=["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"][: len(names)])
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}" if isinstance(v, float) else str(v),
                ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def step_env():
    print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU {torch.cuda.get_device_name(0)}")


def step_prepare():
    bundle = load_data(64)
    _bar("类别分布", ["负面", "正面"], [bundle.train_labels.count(0), bundle.train_labels.count(1)],
         FIG_DIR / "01_类别分布.png", "样本数")
    _bar("数据划分", ["训练", "验证", "测试"], list(bundle.sizes), FIG_DIR / "02_数据划分.png", "样本数")
    print(f"训练{bundle.sizes[0]} 验证{bundle.sizes[1]} 测试{bundle.sizes[2]}")


def step_train():
    for exp, cfg in LSTM_EXPS.items():
        print(f"\n--- LSTM {exp} ---")
        bundle = load_data(cfg["batch_size"])
        model = BiLSTMClassifier(len(bundle.vocab), dropout=cfg["dropout"])
        run_training(model, bundle.train_loader, bundle.val_loader, exp,
                     epochs=LSTM_EPOCHS, lr=cfg["lr"], wd=cfg["weight_decay"])

    print("\n--- Transformer ---")
    bundle = load_data(64)
    model = TransformerClassifier(len(bundle.vocab))
    run_training(model, bundle.train_loader, bundle.val_loader, "transformer",
                 epochs=TRANS_EPOCHS, lr=1e-3)


def step_profile():
    rows = []
    bundle = load_data(64)
    for amp in (False, True):
        for name, model in [("Bi-LSTM", BiLSTMClassifier(len(bundle.vocab))), ("Transformer", TransformerClassifier(len(bundle.vocab)))]:
            r = run_training(model, bundle.train_loader, bundle.val_loader,
                             f"profile_{name}_{'amp' if amp else 'fp32'}",
                             epochs=PROFILE_EPOCHS, lr=1e-3, amp=amp, profile_only=True)
            rows.append({"model": name, "amp": "AMP" if amp else "FP32",
                         "sec": r["total_sec"], "mem": r["peak_mem_mb"]})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "profile.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["model", "amp", "sec", "mem"])
        w.writeheader()
        w.writerows(rows)
    _bar("训练耗时(秒)", [f"{r['model']}-{r['amp']}" for r in rows], [r["sec"] for r in rows],
         FIG_DIR / "05_Profile耗时.png", "秒")
    print("Profile 完成")


def step_eval():
    bundle = load_data(64)
    results = {}
    for exp, dropout in [("exp4_regularized", 0.5), ("exp1_baseline", 0.3)]:
        ckpt = CKPT_DIR / f"{exp}_best.pth"
        if ckpt.exists():
            results["lstm"] = test_model(BiLSTMClassifier(len(bundle.vocab), dropout=dropout), bundle.test_loader, ckpt)
            break
    ckpt = CKPT_DIR / "transformer_best.pth"
    if ckpt.exists():
        results["transformer"] = test_model(TransformerClassifier(len(bundle.vocab)), bundle.test_loader, ckpt)
    if results:
        _bar("测试准确率", list(results.keys()), list(results.values()), FIG_DIR / "04_模型对比.png", "准确率")
        (RESULTS_DIR / "test_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"测试集 {results}")


def step_figures():
    names, accs = [], []
    for exp in LSTM_EXPS:
        p = RESULTS_DIR / f"{exp}.json"
        if p.exists():
            names.append(exp)
            accs.append(json.loads(p.read_text(encoding="utf-8"))["best_val_acc"])
    if names:
        _bar("LSTM超参实验", names, accs, FIG_DIR / "03_超参对比.png", "验证准确率")


def step_summary():
    lines = ["| 实验 | lr | batch | dropout | 验证Acc |", "|---|---:|---:|---:|---:|"]
    for exp, cfg in LSTM_EXPS.items():
        p = RESULTS_DIR / f"{exp}.json"
        if p.exists():
            acc = json.loads(p.read_text(encoding="utf-8"))["best_val_acc"]
            lines.append(f"| {exp} | {cfg['lr']} | {cfg['batch_size']} | {cfg['dropout']} | {acc:.4f} |")
    p = RESULTS_DIR / "transformer.json"
    if p.exists():
        acc = json.loads(p.read_text(encoding="utf-8"))["best_val_acc"]
        lines.append(f"| transformer | 1e-3 | 64 | 0.3 | {acc:.4f} |")
    text = "\n".join(lines)
    (RESULTS_DIR / "实验记录表.md").write_text("# 实验记录\n\n" + text, encoding="utf-8")
    print(text)


def main():
    steps = [("环境", step_env), ("数据", step_prepare), ("训练", step_train),
             ("Profile", step_profile), ("评估", step_eval), ("图表", step_figures), ("汇总", step_summary)]
    print("一键训练开始（完整 7 步，约 10~30 分钟）")
    for i, (name, fn) in enumerate(steps, 1):
        print(f"\n{'=' * 50}\n[{i}/{len(steps)}] {name}")
        t0 = time.time()
        try:
            fn()
            print(f"完成，耗时 {time.time() - t0:.1f}s")
        except Exception:
            import traceback
            traceback.print_exc()
            sys.exit(1)
    print("\n全部完成。模型 outputs/checkpoints/  演示请运行 run_demo.py")


if __name__ == "__main__":
    main()
