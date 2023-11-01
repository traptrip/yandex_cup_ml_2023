import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_uniform(m.weight)
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        m.bias.data.fill_(0.01)
