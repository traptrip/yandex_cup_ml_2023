"""
https://github.com/ShadeAlsha/LTR-weight-balancing
"""

import torch


class Normalizer:
    def __init__(self, LpNorm=2, tau=1):
        self.LpNorm = LpNorm
        self.tau = tau

    def apply_on(
        self, model
    ):  # this method applies tau-normalization on the classifier layer
        for curLayer in [model.classifier[-1].weight]:
            curparam = curLayer.data

            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = (
                (torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1) ** self.tau)
                .detach()
                .unsqueeze(-1)
            )
            scalingVect = torch.ones_like(curparam)

            idx = neuronNorm_curparam == neuronNorm_curparam
            idx = idx.squeeze()
            tmp = 1 / (neuronNorm_curparam[idx].squeeze())
            for _ in range(len(scalingVect.shape) - 1):
                tmp = tmp.unsqueeze(-1)

            scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
            curparam[idx] = scalingVect[idx] * curparam[idx]
