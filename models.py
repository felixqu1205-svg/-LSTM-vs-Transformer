import torch
import torch.nn as nn

from config import MAX_LEN


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        return self.fc(self.drop(out.max(dim=1).values)).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, layers=2, max_len=MAX_LEN, dropout=0.3):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 256, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x[:, : self.max_len]
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        h = self.emb(x) + self.pos(pos)
        pad = x == 0
        h = self.encoder(h, src_key_padding_mask=pad)
        mask = (~pad).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(pooled)).squeeze(-1)
