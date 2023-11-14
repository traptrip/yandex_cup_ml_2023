import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation):
        super().__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, X, H_prev, C_prev):
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)
        input_gate = torch.sigmoid(i_conv)
        forget_gate = torch.sigmoid(f_conv)
        output_gate = torch.sigmoid(o_conv)
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)
        H = output_gate * self.activation(C)
        return H, C


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation):
        super().__init__()
        self.out_channels = out_channels
        self.convLSTMCell = ConvLSTMCell(
            in_channels, out_channels, kernel_size, padding, activation
        )

    def forward(self, X):
        batch_size, seq_len, _, height, width = X.size()
        output = torch.zeros(
            batch_size,
            seq_len,
            self.out_channels,
            height,
            width,
            device=self.convLSTMCell.conv.weight.device,
        )
        H = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width,
            device=self.convLSTMCell.conv.weight.device,
        )
        C = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width,
            device=self.convLSTMCell.conv.weight.device,
        )
        for time_step in range(seq_len):
            H, C = self.convLSTMCell(X[:, time_step], H, C)
            output[:, time_step] = H
        return output


class Seq2Seq(nn.Module):
    def __init__(
        self,
        num_channels,
        num_kernels,
        kernel_size,
        padding,
        activation,
        num_layers,
        out_seq_len,
    ):
        super().__init__()
        self.out_seq_len = out_seq_len

        self.sequential = nn.Sequential()
        self.sequential.add_module(
            "convlstm1",
            ConvLSTM(
                in_channels=num_channels,
                out_channels=num_kernels,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
            ),
        )
        for layer_index in range(2, num_layers + 1):
            self.sequential.add_module(
                f"convlstm{layer_index}",
                ConvLSTM(
                    in_channels=num_kernels,
                    out_channels=num_kernels,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=activation,
                ),
            )
        self.conv = nn.Conv2d(
            in_channels=num_kernels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, X, mask=None):
        batch_size, seq_len, num_channels, height, width = X.size()
        inputs = torch.zeros(
            batch_size,
            seq_len + self.out_seq_len - 1,
            num_channels,
            height,
            width,
            device=self.conv.weight.device,
        )
        inputs[:, :seq_len] = X
        output = self.sequential(inputs)
        output = torch.stack(
            [
                self.conv(output[:, index + seq_len - 1])
                for index in range(self.out_seq_len)
            ],
            dim=1,
        )
        return output


# class ConvLSTMModel(L.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.model = Seq2Seq(
#             num_channels=1,
#             num_kernels=32,
#             kernel_size=(3, 3),
#             padding=(1, 1),
#             activation="relu",
#             num_layers=1,
#             out_seq_len=12,
#         )

#     def forward(self, x):
#         x = x.to(device=self.model.conv.weight.device)
#         output = self.model(x)
#         return output

#     def training_step(self, batch):
#         x, y = batch
#         out = self.forward(x)
#         out[y == -1] = -1
#         loss = F.mse_loss(out, y)
#         self.log("train_loss", loss)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
#         return optimizer


if __name__ == "__main__":
    device = "cuda:1"
    batch, seq_len, n_channels, height, width = 1, 16, 1, 252, 252
    features = torch.zeros(1, seq_len, n_channels, height, width).to(device)
    net = Seq2Seq(
        num_channels=n_channels,
        num_kernels=32,
        kernel_size=(3, 3),
        padding=(1, 1),
        activation="relu",
        num_layers=1,
        out_seq_len=12,
    ).to(device)
