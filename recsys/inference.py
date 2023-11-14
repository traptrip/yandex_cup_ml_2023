import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.dataset import EmbeddingDataset, Collator


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
            _, pred_logits = net(batch, mask)
            pred_probs = torch.sigmoid(pred_logits)

            predictions.append(pred_probs.cpu().numpy())
            track_idxs.append(track_ids.numpy())
    track_idxs = np.concatenate(track_idxs)
    predictions = np.concatenate(predictions)
    return track_idxs, predictions


def run(cfg, net, max_crop_size, is_fixed_crop, test_dataloader=None):
    if not test_dataloader:
        test_dataset = EmbeddingDataset(
            cfg.data_dir, cfg.meta_info, cfg.num_labels, stage="infer"
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=Collator(
                "test",
                cfg.num_labels,
                mix_proba=0,
                max_crop_size=max_crop_size,
                is_fixed_crop=is_fixed_crop,
            ),
        )

    net.to(cfg.device)

    track_idxs, predictions = predict(net, test_dataloader, cfg.device)
    return track_idxs, predictions


if __name__ == "__main__":
    run()
