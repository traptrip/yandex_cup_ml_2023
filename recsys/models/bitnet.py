import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

from .pooling import MaskedPooling, AttentionPooling
from .utils import RMSNorm, RotaryEmbedding, apply_rotary_pos_emb
from utils.nn_utils import init_weights


def absmax_quantize(x, bits=8):
    Qb = 2 ** (bits - 1) - 1
    scale = Qb / torch.max(torch.abs(x))
    quant = (scale * x).round()
    dequant = quant / scale
    return quant.to(torch.int8), dequant


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, groups=1):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        # Group Quantization and Normalization
        weight = self.weight.view(self.groups, -1)

        # binarize weight
        weight = weight - weight.mean(dim=1, keepdim=True)
        weight = torch.sign(weight)

        # scaling factor
        beta = torch.abs(weight).sum(dim=1, keepdim=True) / (
            weight.shape[0] * weight.shape[1]
        )

        weight = weight * beta
        weight = weight.view(self.out_features, self.in_features)

        # Absmax Quantization
        quant_input, _ = absmax_quantize(input)

        # Linear
        output = F.linear(quant_input.float(), weight)

        # Dequantization
        output = output / beta.view(-1, 1)
        return output


# all we need
class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = RMSNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        # self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.fused_attn_ff_proj = BitLinear(dim, sum(self.fused_dims))

        # self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        self.attn_out = BitLinear(attn_inner_dim, dim)

        # Swap out the linear layers here
        # self.ff_out = nn.Sequential(nn.GELU(), nn.Linear(ff_inner_dim, dim, bias=False))
        self.ff_out = nn.Sequential(nn.GELU(), BitLinear(ff_inner_dim, dim))

        # for caching causal mask and rotary embeddings
        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        ff_mult=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                ParallelTransformerBlock(dim, dim_head, heads, ff_mult),
            )

    def forward(self, x):
        for block in self.layers:
            x = block(x) + x
        return x


class Network(nn.Module):
    def __init__(
        self,
        transformer_layers: int,
        num_heads: int = 8,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_labels: int = 256,
        pooling: str = "attention",
    ):
        super().__init__()
        self.num_labels = num_labels
        self.input_dim = input_dim
        assert pooling in ("attention", "mask")

        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         input_dim,
        #         num_heads,
        #         dim_feedforward=2048,
        #         dropout=0.2,
        #         batch_first=True,
        #     ),
        #     num_layers=transformer_layers,
        # )
        self.transformer = Transformer(input_dim, transformer_layers, num_heads, 2048)

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
        x = self.transformer(x)
        x = self.pooling(x, mask).squeeze()
        x = self.projector(x)
        out = self.classifier(x)
        return torch.nn.functional.normalize(x, p=2, dim=1), out
