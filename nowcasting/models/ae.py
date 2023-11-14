# import torch
# import numpy as np
# import torch.nn as nn

# from ldcast.ldcast.models.blocks.resnet import ResBlock3D
# from ldcast.ldcast.models.utils import activation, normalization
# from ldcast.ldcast.models.distributions import (
#     kl_from_standard_normal,
#     ensemble_nll_normal,
#     sample_from_standard_normal,
# )


# class SimpleConvEncoder(nn.Sequential):
#     def __init__(self, in_dim=1, levels=2, min_ch=64):
#         sequence = []
#         channels = np.hstack([in_dim, (8 ** np.arange(1, levels + 1)).clip(min=min_ch)])

#         for i in range(levels):
#             in_channels = int(channels[i])
#             out_channels = int(channels[i + 1])
#             res_kernel_size = (3, 3, 3) if i == 0 else (1, 3, 3)
#             res_block = ResBlock3D(
#                 in_channels,
#                 out_channels,
#                 kernel_size=res_kernel_size,
#                 norm_kwargs={"num_groups": 1},
#             )
#             sequence.append(res_block)
#             downsample = nn.Conv3d(
#                 out_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)
#             )
#             sequence.append(downsample)
#             in_channels = out_channels

#         super().__init__(*sequence)


# class SimpleConvDecoder(nn.Sequential):
#     def __init__(self, in_dim=1, levels=2, min_ch=64):
#         sequence = []
#         channels = np.hstack([in_dim, (8 ** np.arange(1, levels + 1)).clip(min=min_ch)])

#         for i in reversed(list(range(levels))):
#             in_channels = int(channels[i + 1])
#             out_channels = int(channels[i])
#             upsample = nn.ConvTranspose3d(
#                 in_channels, in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)
#             )
#             sequence.append(upsample)
#             res_kernel_size = (3, 3, 3) if (i == 0) else (1, 3, 3)
#             res_block = ResBlock3D(
#                 in_channels,
#                 out_channels,
#                 kernel_size=res_kernel_size,
#                 norm_kwargs={"num_groups": 1},
#             )
#             sequence.append(res_block)
#             in_channels = out_channels

#         super().__init__(*sequence)


# class AutoencoderKL(nn.Module):
#     def __init__(
#         self,
#         encoder,
#         decoder,
#         kl_weight=0.01,
#         encoded_channels=64,
#         hidden_width=32,
#         out_seq_len=12,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.out_seq_len = out_seq_len
#         self.encoder = encoder
#         self.decoder = decoder
#         self.hidden_width = hidden_width
#         self.to_moments = nn.Conv3d(encoded_channels, 2 * hidden_width, kernel_size=1)
#         self.to_decoder = nn.Conv3d(hidden_width, encoded_channels, kernel_size=1)
#         self.log_var = nn.Parameter(torch.zeros(size=()))
#         self.kl_weight = kl_weight

#     def encode(self, x):
#         h = self.encoder(x)
#         (mean, log_var) = torch.chunk(self.to_moments(h), 2, dim=1)
#         return (mean, log_var)

#     def decode(self, z):
#         z = self.to_decoder(z)
#         dec = self.decoder(z)
#         return dec

#     def forward(self, x, sample_posterior=True):
#         batch_size, seq_len, num_channels, height, width = x.size()
#         x = x.permute(0, 2, 1, 3, 4)
#         inputs = torch.zeros(
#             batch_size,
#             num_channels,
#             # seq_len + self.out_seq_len - 1,
#             self.out_seq_len,
#             height,
#             width,
#             device=self.to_moments.weight.device,
#         )
#         inputs[:, :, :seq_len] = x

#         (mean, log_var) = self.encode(inputs)
#         if sample_posterior:
#             z = sample_from_standard_normal(mean, log_var)
#         else:
#             z = mean
#         dec = self.decode(z)
#         dec = dec.permute(0, 2, 1, 3, 4)
#         return (dec, mean, log_var)

#     def loss(self, y_pred, y_true):
#         rec_loss = (y_true - y_pred).abs().mean()
#         kl_loss = kl_from_standard_normal(mean, log_var)

#         total_loss = rec_loss + self.kl_weight * kl_loss

#         return (total_loss, rec_loss, kl_loss)
