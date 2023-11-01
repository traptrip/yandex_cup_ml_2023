import torch
import torch.nn as nn

from ..utils.nn_utils import init_weights


class Network(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_labels=256):
        super().__init__()
        self.num_labels = num_labels
        self.bn = nn.LayerNorm(hidden_dim)
        self.projector = nn.Linear(input_dim, hidden_dim)
        self.lin = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
        )
        self.projector.apply(init_weights)
        self.lin.apply(init_weights)
        self.classifier.apply(init_weights)
    
    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        x = [self.projector(x[~m]) for x, m in zip(features, mask)]  # 768 -> 512
        x = torch.cat([v.mean(0).unsqueeze(0) for v in x], 0)
        x = self.bn(x)
        x = self.lin(x)
        outs = self.classifier(x)
        return outs
