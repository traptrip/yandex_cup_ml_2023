from functools import partial

import torch
import torch.nn as nn

from .pooling import MaskedPooling, AttentionPooling
from utils.nn_utils import init_weights


class Network(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        crop_size: int = 80,
        num_labels: int = 256,
        pooling: str = "attention",
    ):
        super().__init__()
        self.num_labels = num_labels
        self.input_dim = input_dim
        self.crop_size = crop_size
        assert pooling in ("attention", "mask", "none")

        # b,crop_size,512 -> b,1,256
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.conv1 = nn.Conv1d(
            crop_size, 1, kernel_size=3, stride=2, dilation=1, padding=1
        )

        # b,crop_size,768 -> b,1,256
        # self.conv1 = nn.Conv1d(
        #     crop_size, 1, kernel_size=5, stride=3, dilation=1, padding=1
        # )

        if pooling == "mask":
            self.pooling = MaskedPooling()
        elif pooling == "attention":
            self.pooling = AttentionPooling(input_dim)
        elif pooling == "none":
            self.pooling = nn.Identity()

        self.classifier = nn.Identity()

    def forward(self, x, mask=None):
        x = self.proj(x)
        x = self.conv1(x).squeeze()
        # x = self.pooling(x, mask).squeeze()
        out = self.classifier(x)
        return torch.nn.functional.normalize(x, p=2, dim=1), out
