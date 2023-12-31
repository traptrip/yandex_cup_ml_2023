import torch
import torch.nn as nn


def init_weights(m, dtype="kaiming"):
    if isinstance(m, nn.Linear):
        if dtype == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif dtype == "kaiming":
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        m.bias.data.fill_(0.01)


def create_label_matrix(labels: torch.Tensor, other_labels=None):
    labels = labels.squeeze()

    if labels.ndim == 1:
        if other_labels is None:
            return (labels.view(-1, 1) == labels.t()).float()
        return (labels.view(-1, 1) == other_labels.t()).float()

    elif labels.ndim == 2:
        size = labels.size(0)
        if other_labels is None:
            return (labels.view(size, size, 1) == labels.view(size, 1, size)).float()
        raise NotImplementedError(
            f"Function for tensor dimension {labels.ndim} comparated to tensor of dimension {other_labels.ndim} not implemented"
        )

    raise NotImplementedError(
        f"Function for tensor dimension {labels.ndim} not implemented"
    )
