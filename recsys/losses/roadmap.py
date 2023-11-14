import torch.nn as nn

from .calibration_loss import CalibrationLoss
from .smooth_rank_ap import SupAP


class RoadmapLoss(nn.Module):
    """
    - name: CalibrationLoss
    weight: 1.0
    kwargs:
        pos_margin: 0.9
        neg_margin: 0.6

    - name: SupAP
    weight: 1.0
    kwargs:
        tau: 0.01
        rho: 100.0
        offset: 1.44
        delta: 0.05
    """

    def __init__(
        self,
        weights: list[float],
        calibration_args: dict[str, float],
        sup_ap_args: dict[str, float],
    ) -> None:
        super().__init__()
        self.w = weights
        self.contrastive_loss = CalibrationLoss(**calibration_args)
        self.sup_ap_loss = SupAP(**sup_ap_args)

    def forward(self, embeddings, targets):
        return self.w[0] * self.contrastive_loss(embeddings, targets) + self.w[
            1
        ] * self.sup_ap_loss(embeddings, targets)
