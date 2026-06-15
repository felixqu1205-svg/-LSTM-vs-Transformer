import json
import random
import re
from collections import Counter
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from config import DATA_DIR, MAX_LEN, SEED, TEST_PER_CLASS, TRAIN_PER_CLASS, VAL_RATIO

CACHE = DATA_DIR / "chinese_sentiment.json"


def clean_text(text):
    """中文清洗：去空白，保留可读字符。"""
    text = re.sub(r"\s+", "", text.strip())
    return text


def tokenize(text):
    return list(clean_text(text))


def build_vocab(texts, max_size=8000):
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, c in cnt.most_common(max_size):
        if c >= 2:
            vocab[ch] = len(vocab)
    return vocab


def encode(text, vocab, max_len=MAX_LEN):
    ids = [vocab.get(ch, 1) for ch in tokenize(text)[:max_len]]
    return ids + [0] * (max_len - len(ids))


def _gen_chinese_reviews(n, label, rng):
    """生成不重复中文影评（字级输入）。"""
    targets = ["这部电影", "这部片子", "这部影片", "这个剧情", "这个故事", "这场表演",
               "这位主演", "这部国产片", "这部悬疑片", "这部爱情片", "这部喜剧", "这部科幻片"]
    pos_adj = ["非常精彩", "十分感人", "特别好看", "相当出色", "令人难忘", "温暖治愈",
               "节奏紧凑", "逻辑清晰", "制作精良", "演技炸裂", "镜头很美", "配乐动听",
               "笑点密集", "高潮迭起", "细节到位", "诚意满满"]
    neg_adj = ["非常无聊", "十分尴尬", "特别难看", "相当失望", "令人困倦", "逻辑混乱",
               "节奏拖沓", "演技浮夸", "剧情老套", "台词生硬", "制作粗糙", "毫无亮点",
               "强行煽情", "结局仓促", "浪费票钱", "看不下去"]
    pos_tail = ["值得推荐", "值得二刷", "我会推荐给朋友", "整体体验很好", "看完心情很好",
                "超出我的预期", "是近期佳作", "让我意犹未尽", "值得认真品味"]
    neg_tail = ["不建议观看", "不会再看了", "纯粹浪费时间", "整体体验很差", "看完很失望",
                "完全不值票价", "是近期烂片", "让我中途想离场", "不值得讨论"]
    actions = ["我觉得", "我认为", "看完感觉", "个人感受是", "老实说", "总体来说"]
    seen, texts = set(), []
    while len(texts) < n:
        t = rng.choice(targets)
        adj = rng.choice(pos_adj if label else neg_adj)
        tail = rng.choice(pos_tail if label else neg_tail)
        tpl = rng.randint(0, 3)
        if tpl == 0:
            s = f"{rng.choice(actions)}，{t}{adj}，{tail}。"
        elif tpl == 1:
            s = f"{t}{adj}，{rng.choice(actions[:-1])}整体{adj}，{tail}。"
        elif tpl == 2:
            s = f"今天看了{t}，{adj}，最后{tail}，编号{rng.randint(1, 999999)}。"
        else:
            s = f"{t}让我{rng.choice(['印象深刻', '眼前一亮', '收获很多'] if label else ['很无语', '很失望', '很难受'])}，{adj}，{tail}。"
        if s not in seen:
            seen.add(s)
            texts.append(s)
    return texts


def ensure_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    need = TRAIN_PER_CLASS * 2
    if CACHE.exists():
        d = json.loads(CACHE.read_text(encoding="utf-8"))
        if d.get("lang") == "zh" and len(d.get("train_labels", [])) >= need:
            return
    rng = random.Random(SEED)
    tr_neg = _gen_chinese_reviews(TRAIN_PER_CLASS, 0, rng)
    tr_pos = _gen_chinese_reviews(TRAIN_PER_CLASS, 1, rng)
    te_neg = _gen_chinese_reviews(TEST_PER_CLASS, 0, rng)
    te_pos = _gen_chinese_reviews(TEST_PER_CLASS, 1, rng)
    pairs = list(zip(tr_neg + tr_pos, [0] * TRAIN_PER_CLASS + [1] * TRAIN_PER_CLASS))
    rng.shuffle(pairs)
    train_texts, train_labels = [p[0] for p in pairs], [p[1] for p in pairs]
    pairs = list(zip(te_neg + te_pos, [0] * TEST_PER_CLASS + [1] * TEST_PER_CLASS))
    rng.shuffle(pairs)
    test_texts, test_labels = [p[0] for p in pairs], [p[1] for p in pairs]
    CACHE.write_text(json.dumps({
        "lang": "zh",
        "train_texts": train_texts, "train_labels": train_labels,
        "test_texts": test_texts, "test_labels": test_labels,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"[数据] 中文数据集：训练 {len(train_labels)} 条，测试 {len(test_labels)} 条")


class TextSet(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts, self.labels, self.vocab = texts, labels, vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        x = torch.tensor(encode(self.texts[i], self.vocab), dtype=torch.long)
        y = torch.tensor(self.labels[i], dtype=torch.float)
        return x, y


@dataclass
class DataBundle:
    vocab: dict
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_labels: list
    sizes: tuple


def load_data(batch_size=64):
    random.seed(SEED)
    torch.manual_seed(SEED)
    ensure_dataset()
    d = json.loads(CACHE.read_text(encoding="utf-8"))
    texts, labels = d["train_texts"], d["train_labels"]
    idx = list(range(len(texts)))
    random.shuffle(idx)
    v = max(1, int(len(texts) * VAL_RATIO))
    tr_i, va_i = idx[v:], idx[:v]
    tr_t, tr_y = [texts[i] for i in tr_i], [labels[i] for i in tr_i]
    va_t, va_y = [texts[i] for i in va_i], [labels[i] for i in va_i]
    vocab = build_vocab(tr_t)
    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    print(f"[数据] 字级词表 {len(vocab)} | 训练 {len(tr_i)} 验证 {len(va_i)} 测试 {len(d['test_texts'])}")
    return DataBundle(
        vocab=vocab,
        train_loader=DataLoader(TextSet(tr_t, tr_y, vocab), shuffle=True, **kw),
        val_loader=DataLoader(TextSet(va_t, va_y, vocab), shuffle=False, **kw),
        test_loader=DataLoader(TextSet(d["test_texts"], d["test_labels"], vocab), shuffle=False, **kw),
        train_labels=tr_y,
        sizes=(len(tr_i), len(va_i), len(d["test_texts"])),
    )
