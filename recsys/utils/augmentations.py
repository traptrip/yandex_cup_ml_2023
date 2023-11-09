import math
import random
from typing import Any

import torch
import numpy as np
import torchvision
from torch import Tensor


class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 1.0,
        alpha: float = 1.0,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(
        self, batch: Tensor, target: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, seq, emb_size)
            target (Tensor): Integer tensor of size (B, )
            mask (Tensor): Boolean tensor of size (B, seq)

        Returns:
            Tensor: Randomly transformed batch.
        """
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()
            mask = mask.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)
        mask_rolled = mask.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        mask = torch.logical_and(mask, mask_rolled)

        return batch, target, mask

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 1.0,
        alpha: float = 1.0,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(
        self, batch: Tensor, target: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, seq, emb_size)
            target (Tensor): Integer tensor of size (B, )
            mask (Tensor): Boolean tensor of size (B, seq)

        Returns:
            Tensor: Randomly transformed batch.
        """
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()
            mask = mask.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)
        mask_rolled = mask.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        _, seq_len, emb_len = torchvision.transforms.functional.get_dimensions(batch)
        H, W = seq_len, emb_len

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = 0  # int(torch.clamp(r_x - r_w_half, min=0))
        x2 = W  # int(torch.clamp(r_x + r_w_half, max=W))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, y1:y2, x1:x2] = batch_rolled[:, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        mask[:, y1:y2] = mask_rolled[:, y1:y2]

        return batch, target, mask

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomAugment:
    def __call__(self, emb: Tensor) -> Tensor:
        if random.random() < 0.5:
            emb = emb.flip(0)

        if random.random() < 0.5:
            cuts = random.randint(1, 8)
            # cuts = random.randint(1, int(len(emb) * 0.4))
            idxs_to_cut = np.random.choice(len(emb), cuts)
            m = torch.ones(len(emb), dtype=torch.bool)
            m[idxs_to_cut] = False
            emb = emb[m]

        return emb
