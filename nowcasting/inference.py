from pathlib import Path

import torch
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score

from utils.dataset import RadarDataset, Collator


def process_test(model, device, test_loader, output_file="prediction.hdf5"):
    model.eval()
    with torch.no_grad():
        for index, item in enumerate(tqdm(test_loader)):
            inputs, last_input_timestamp = item["features"], item["timestamp"]
            output = model(inputs.to(device))
            output = output.cpu().numpy()
            with h5py.File(output_file, mode="a") as f_out:
                for index in range(output.shape[1]):
                    timestamp_out = str(
                        int(last_input_timestamp[-1]) + 600 * (index + 1)
                    )
                    f_out.create_group(timestamp_out)
                    f_out[timestamp_out].create_dataset(
                        "intensity", data=output[0, index, 0]
                    )


def run(cfg, net):
    test_dataset = RadarDataset(cfg.test_files, out_seq_len=0, with_time=True)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        # collate_fn=Collator(
        #     "test",
        #     cfg.num_labels,
        #     mix_proba=0,
        #     max_crop_size=max_crop_size,
        #     is_fixed_crop=is_fixed_crop,
        # ),
    )
    net.to(cfg.device)
    if (cfg.logs_dir / "prediction.hdf5").exists():
        (cfg.logs_dir / "prediction.hdf5").unlink()
    process_test(net, cfg.device, test_dataloader, cfg.logs_dir / "prediction.hdf5")


if __name__ == "__main__":
    run()
