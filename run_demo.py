"""PyCharm 答辩演示：直接运行本文件，输入中文影评即可。"""

import os
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from config import CKPT_DIR, MAX_LEN
from data import clean_text, encode, load_data
from models import BiLSTMClassifier, TransformerClassifier

EXAMPLES = [
    ("正面", "这部电影非常精彩，演技炸裂，值得推荐。"),
    ("负面", "剧情拖沓无聊，演技尴尬，纯粹浪费时间。"),
    ("中性偏正", "整体节奏紧凑，故事还算完整，可以一看。"),
]


class DemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("中文情感分析演示 · LSTM vs Transformer")
        self.root.geometry("720x520")
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ttk.Label(root, text="中文影评情感分析", font=("Microsoft YaHei", 14, "bold")).pack(pady=8)
        self.status = tk.StringVar(value="加载模型中...")
        ttk.Label(root, textvariable=self.status).pack()
        self.input = scrolledtext.ScrolledText(root, height=5, font=("Microsoft YaHei", 12))
        self.input.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self.input.insert(tk.END, EXAMPLES[0][1])

        row = ttk.Frame(root)
        row.pack(pady=4)
        ttk.Button(row, text="分析", command=self.analyze).pack(side=tk.LEFT)
        for label, text in EXAMPLES:
            ttk.Button(row, text=label, command=lambda t=text: self._fill(t)).pack(side=tk.LEFT, padx=6)

        self.output = scrolledtext.ScrolledText(root, height=8, font=("Microsoft YaHei", 12), state=tk.DISABLED)
        self.output.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self.root.after(50, self.load_models)

    def _fill(self, text):
        self.input.delete("1.0", tk.END)
        self.input.insert(tk.END, text)

    def load_models(self):
        try:
            bundle = load_data(32)
            self.vocab = bundle.vocab
            self.lstm = BiLSTMClassifier(len(bundle.vocab), dropout=0.5).to(self.dev)
            self.trans = TransformerClassifier(len(bundle.vocab)).to(self.dev)
            for model, names in [(self.lstm, ["exp4_regularized", "exp1_baseline"]), (self.trans, ["transformer"])]:
                for n in names:
                    p = CKPT_DIR / f"{n}_best.pth"
                    if p.exists():
                        model.load_state_dict(torch.load(p, map_location=self.dev, weights_only=True))
                        break
                else:
                    raise FileNotFoundError("请先运行 run_train.py 完成中文数据训练")
            self.status.set(f"就绪 | {self.dev}")
            self.analyze()
        except Exception as e:
            messagebox.showerror("错误", f"{e}\n请先运行 run_train.py")

    def predict(self, model, text):
        ids = torch.tensor([encode(clean_text(text), self.vocab, MAX_LEN)], device=self.dev)
        with torch.no_grad():
            p_pos = torch.sigmoid(model(ids)).item()
        label = "正面" if p_pos > 0.5 else "负面"
        conf = p_pos if label == "正面" else 1 - p_pos
        return label, conf

    def analyze(self):
        if not hasattr(self, "lstm"):
            return
        text = self.input.get("1.0", tk.END).strip()
        if not text:
            return
        la, pa = self.predict(self.lstm, text)
        lt, pt = self.predict(self.trans, text)
        msg = (
            f"输入：{text}\n\n"
            f"LSTM：{la}（置信度 {pa:.1%}）\n"
            f"Transformer：{lt}（置信度 {pt:.1%}）"
        )
        self.output.configure(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, msg)
        self.output.configure(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()
