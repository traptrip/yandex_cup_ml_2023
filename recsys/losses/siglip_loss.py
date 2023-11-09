import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pytorch_metric_learning import losses, distances
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


# class SigLIPLoss(nn.Module):
#     """Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

#     @article{zhai2023sigmoid,
#       title={Sigmoid loss for language image pre-training},
#       author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
#       journal={arXiv preprint arXiv:2303.15343},
#       year={2023}
#     }

#     # Pseudo code
#     # img_emb : image model embedding [n, dim]
#     # txt_emb : text model embedding [n, dim]
#     # t_prime, b : learnable temperature and bias
#     # n : mini-batch size
#     t = exp(t_prime)
#     zimg = l2_normalize(img_emb)
#     ztxt = l2_normalize(txt_emb)
#     logits = dot(zimg, ztxt.T) * t + b
#     labels = 2 * eye(n) - ones(n) # -1 with diagonal 1
#     l = -sum(log_sigmoid(labels * logits)) / n
#     """

#     def __init__(self, logit_scale=1, logit_bias=0.01):
#         super().__init__()
#         self.logit_scale = nn.Parameter(torch.tensor(logit_scale))
#         self.logit_bias = nn.Parameter(torch.tensor(logit_bias))

#     def get_labels(self, num_logits, device, dtype) -> torch.Tensor:
#         labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
#         labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
#         return labels

#     def get_logits(self, emb1: Tensor, emb2: Tensor):
#         return (emb1 @ emb2.T) * torch.exp(self.logit_scale) + self.logit_bias

#     def _compute_loss(self, emb1: Tensor, emb2: Tensor):
#         logits = self.get_logits(emb1, emb2)
#         labels = self.get_labels(emb1.device, emb1.dtype, emb1.shape[0])
#         loss = -F.logsigmoid(labels * logits).sum() / emb1.shape[0]
#         return loss

#     def forward(self, emb1: Tensor, emb2: Tensor):
#         # emb1 & emb2 should be L2 normalized!
#         loss = self._compute_loss(emb1, emb2)
#         return loss


class LogSigSimoidilarity(distances.BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def forward(self, query_emb, ref_emb=None, scale=None, bias=None):
        self.reset_stats()
        self.check_shapes(query_emb, ref_emb)
        query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb)
        self.set_default_stats(
            query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
        )
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized, scale, bias)
        if self.power != 1:
            mat = mat**self.power
        assert mat.size() == torch.Size((query_emb.size(0), ref_emb.size(0)))
        return mat

    def compute_mat(
        self, query_emb: Tensor, ref_emb: Tensor, scale: Tensor, bias: Tensor
    ):
        mat = torch.matmul(query_emb, ref_emb.t()) * torch.exp(scale) + bias
        return mat

    def pairwise_distance(self, query_emb: Tensor, ref_emb: Tensor):
        return torch.sum(F.logsigmoid(query_emb * ref_emb), dim=1)


class SigLIPLoss(losses.ContrastiveLoss):
    def __init__(self, logit_scale=1.0, logit_bias=0.01, **kwargs):
        super().__init__(**kwargs)
        self.scale = nn.Parameter(torch.tensor(logit_scale))
        self.bias = nn.Parameter(torch.tensor(logit_bias))

    def get_default_distance(self):
        return LogSigSimoidilarity()

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels, ref_labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb, self.scale, self.bias)
        return self.loss_method(mat, indices_tuple)

    def _compute_loss(self, pos_pair_dist, neg_pair_dist, indices_tuple):
        pos_loss, neg_loss = 0, 0
        if len(pos_pair_dist) > 0:
            pos_loss = self.get_per_pair_loss(pos_pair_dist, "pos")
        if len(neg_pair_dist) > 0:
            neg_loss = self.get_per_pair_loss(neg_pair_dist, "neg")
        pos_pairs = lmu.pos_pairs_from_tuple(indices_tuple)
        neg_pairs = lmu.neg_pairs_from_tuple(indices_tuple)
        return {
            "pos_loss": {
                "losses": pos_loss,
                "indices": pos_pairs,
                "reduction_type": "pos_pair",
            },
            "neg_loss": {
                "losses": neg_loss,
                "indices": neg_pairs,
                "reduction_type": "neg_pair",
            },
        }

    def get_per_pair_loss(self, pair_dists, pos_or_neg):
        loss_calc_func = self.pos_calc if pos_or_neg == "pos" else self.neg_calc
        margin = self.pos_margin if pos_or_neg == "pos" else self.neg_margin
        per_pair_loss = loss_calc_func(pair_dists, margin)
        return per_pair_loss

    def pos_calc(self, pos_pair_dist, margin):
        return F.logsigmoid(self.distance.margin(pos_pair_dist, margin))

    def neg_calc(self, neg_pair_dist, margin):
        return F.logsigmoid(self.distance.margin(margin, neg_pair_dist))

    def forward(
        self, embeddings: Tensor, labels: Tensor, ref_embeddings=None, ref_labels=None
    ):
        if ref_embeddings is None:
            return super().forward(embeddings, labels)

        indices_tuple = self.create_indices_tuple(
            embeddings.size(0),
            embeddings,
            labels,
            ref_embeddings,
            ref_labels,
        )

        combined_embeddings = torch.cat([embeddings, ref_embeddings], dim=0)
        combined_labels = torch.cat([labels, ref_labels], dim=0)
        return super().forward(combined_embeddings, combined_labels, indices_tuple)

    def create_indices_tuple(
        self,
        batch_size,
        embeddings,
        labels,
        E_mem,
        L_mem,
    ):
        indices_tuple = lmu.get_all_pairs_indices(labels, L_mem)
        indices_tuple = c_f.shift_indices_tuple(indices_tuple, batch_size)
        return indices_tuple


if __name__ == "__main__":
    criterion = SigLIPLoss()

    embeddings = torch.rand((4, 768))
    labels = torch.tensor([1, 1, 2, 2])
    loss = criterion(embeddings, labels)
    print(loss)
