import torch
import torch.nn as nn

from .pooling import MaskedPooling


class Network(nn.Module):
    def __init__(
        self,
        transformer_layers: int,
        num_heads: int = 8,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_labels: int = 256,
    ):
        super().__init__()
        self.num_labels = num_labels

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                input_dim,
                num_heads,
                dim_feedforward=2048,
                dropout=0.2,
                batch_first=True,
            ),
            num_layers=transformer_layers,
        )
        # self.pooling = nn.AdaptiveAvgPool2d((1, input_dim))
        self.pooling = MaskedPooling()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x, mask=None):
        x = self.transformer(x)
        x = self.pooling(x, mask).squeeze()
        x = self.projector(x)
        return x
