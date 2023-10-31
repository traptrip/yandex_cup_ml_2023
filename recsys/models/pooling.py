import torch
import torch.nn as nn


class MaskedPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is not None:
            x = torch.cat([v[m].mean(0).unsqueeze(0) for v, m in zip(x, mask)])
        else:
            x = torch.mean(x, 1)
        return x
