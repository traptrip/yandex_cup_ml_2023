from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score

from utils.dataset import EmbeddingDataset, Collator

# from models.transformer_encoder import Network
from models.mlp import Network


class Config:
    weights: Path = Path("_EXPERIMENTS/mlp_1/weights/best.pt")

    # data
    data_dir = Path("./data/")
    num_labels = 256
    crop_size = None

    batch_size = 160
    num_workers = 4

    # net
    transformer_layers = 1
    num_heads = 8
    input_dim = 768
    hidden_dim = 512

    device = "cuda:1"


def predict(net, loader, device):
    net.eval()
    track_idxs = []
    predictions = []
    with torch.no_grad():
        for data in tqdm(loader):
            track_ids, batch, mask = (
                data["track"],
                data["features"],
                data["mask"],
            )
            batch = batch.to(device)
            mask = mask.to(device)
            pred_logits = net(batch, mask)
            pred_probs = torch.sigmoid(pred_logits)
            pred_probs = torch.round(pred_probs, decimals=6)
            predictions.append(pred_probs.cpu().numpy())
            track_idxs.append(track_ids.numpy())
    track_idxs = np.concatenate(track_idxs)
    predictions = np.concatenate(predictions)
    return track_idxs, predictions


# def predict(net, loader, device):
#     net.eval()
#     track_idxs = []
#     predictions = []
#     targets = []

#     with torch.no_grad():
#         for data in tqdm(loader):
#             track_ids, batch, mask, target = (
#                 data["track"],
#                 data["features"],
#                 data["mask"],
#                 data["label"],
#             )
#             batch = batch.to(device)
#             mask = mask.to(device)
#             pred_logits = net(batch, mask)
#             pred_probs = torch.sigmoid(pred_logits)
#             pred_probs = torch.round(pred_probs, decimals=6)
#             predictions.append(pred_probs.cpu().numpy())
#             track_idxs.append(track_ids.numpy())
#             targets.append(target.numpy())
#     track_idxs = np.concatenate(track_idxs)
#     predictions = np.concatenate(predictions)
#     targets = np.concatenate(targets)

#     print("AP:", average_precision_score(targets, predictions))
#     return track_idxs, predictions


def main():
    cfg = Config()

    test_dataset = EmbeddingDataset(
        cfg.data_dir, cfg.num_labels, cfg.crop_size, stage="infer"
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=Collator("test", cfg.num_labels),
    )

    net = Network(
        # transformer_layers=cfg.transformer_layers,
        # num_heads=cfg.num_heads,
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_labels=cfg.num_labels,
    )
    net.load_state_dict(torch.load(cfg.weights, map_location="cpu"))
    net.to(cfg.device)

    track_idxs, predictions = predict(net, test_dataloader, cfg.device)

    print("Saving prediction")
    predictions_df = pd.DataFrame(
        [
            {"track": track, "prediction": ",".join([str(p) for p in probs])}
            for track, probs in zip(track_idxs, predictions)
        ]
    )
    predictions_df.to_csv(cfg.weights.parent / "../prediction.csv", index=False)


if __name__ == "__main__":
    main()
