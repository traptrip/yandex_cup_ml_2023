import torch
import torch.nn as nn


def init_weights(m, dtype="kaiming"):
    if isinstance(m, nn.Linear):
        if dtype == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif dtype == "kaiming":
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        m.bias.data.fill_(0.01)
