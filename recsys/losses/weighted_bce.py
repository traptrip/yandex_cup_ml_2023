from pathlib import Path

import torch.nn as nn
from torch import Tensor

CUR_DIR = Path(__file__).parent


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, device, weights, **bce_kwargs) -> None:
        super().__init__()
        if "reduction" in bce_kwargs:
            del bce_kwargs["reduction"]
        self.bce = nn.BCEWithLogitsLoss(reduction="none", **bce_kwargs)
        self.weights = weights.to(device)
        self.weights_sum = self.weights.sum()

    def forward(self, input: Tensor, target: Tensor):
        loss = self.bce(input, target) * self.weights
        return loss.sum() / self.weights_sum
