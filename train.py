import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import CKPT_DIR, LOG_DIR, RESULTS_DIR


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits = model(x)
        loss_sum += crit(logits, y).item() * x.size(0)
        correct += ((torch.sigmoid(logits) > 0.5).long() == y.long()).sum().item()
        n += x.size(0)
    return loss_sum / n, correct / n


def train_epoch(model, loader, opt, dev, amp):
    model.train()
    crit = nn.BCEWithLogitsLoss()
    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = crit(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss_sum += loss.item() * x.size(0)
        correct += ((torch.sigmoid(logits) > 0.5).long() == y.long()).sum().item()
        n += x.size(0)
    return loss_sum / n, correct / n


def run_training(model, train_loader, val_loader, name, epochs=10, lr=1e-3, wd=0.0, amp=True, profile_only=False):
    dev = device()
    model = model.to(dev)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    writer = SummaryWriter(LOG_DIR / name)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0, best, best_path = time.perf_counter(), 0.0, CKPT_DIR / f"{name}_best.pth"

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, dev, amp)
        va_loss, va_acc = evaluate(model, val_loader, dev)
        writer.add_scalars("Loss", {"train": tr_loss, "val": va_loss}, ep)
        writer.add_scalars("Accuracy", {"train": tr_acc, "val": va_acc}, ep)
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), best_path)
        print(f"[{name}] epoch {ep}/{epochs} val_acc={va_acc:.4f}")

    total = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**2 if dev.type == "cuda" else 0
    writer.close()
    result = {"run_name": name, "best_val_acc": best, "total_sec": total, "peak_mem_mb": round(peak, 1), "amp": amp}
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


@torch.no_grad()
def test_model(model, loader, ckpt: Path):
    dev = device()
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    model.to(dev)
    return evaluate(model, loader, dev)[1]
