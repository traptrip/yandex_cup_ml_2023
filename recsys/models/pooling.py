import torch
import torch.nn as nn


class MaskedPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is not None:
            x = torch.cat([v[~m].mean(0).unsqueeze(0) for v, m in zip(x, mask)])
        else:
            x = torch.mean(x, 1)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 1),
        )

    def forward(self, x, mask=None):
        attn_logits = self.attn(x)
        if mask is not None:
            attn_logits[mask] = -float("inf")
        attn_weights = torch.softmax(attn_logits, dim=1)
        x = x * attn_weights
        # x = self.dropout(x)
        x = x.sum(dim=1)
        return x
