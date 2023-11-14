from functools import partial

import torch
import torch.nn as nn
from utils.nn_utils import init_weights

from .pooling import AttentionPooling, MaskedPooling


class Network(nn.Module):
    def __init__(
        self,
        transformer_layers: int,
        num_heads: int = 8,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_labels: int = 256,
        pooling: str = "attention",
        dim_feedforward: int = 2048,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.input_dim = input_dim
        assert pooling in ("attention", "mask")

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                input_dim,
                num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=transformer_layers,
        )

        if pooling == "mask":
            self.pooling = MaskedPooling()
        elif pooling == "attention":
            self.pooling = AttentionPooling(input_dim)

        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
        )

        self.projector.apply(init_weights)
        self.classifier.apply(partial(init_weights, dtype="xavier"))

    def forward(self, x, mask=None):
        # x = self.transformer(x)  # use w/o mask if is_fixed_crop=True
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.pooling(x, mask).squeeze()
        x = self.projector(x)
        out = self.classifier(x)
        return torch.nn.functional.normalize(x, p=2, dim=1), out
