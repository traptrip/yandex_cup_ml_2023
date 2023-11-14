import random
import json
from pathlib import Path

import h5py
import torch
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn import preprocessing

from .augmentations import RandomCutmix, RandomMixup


class RadarDataset(Dataset):
    def __init__(
        self,
        list_of_files,
        in_seq_len=4,
        out_seq_len=12,
        mode="overlap",
        with_time=False,
    ):
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.seq_len = in_seq_len + out_seq_len
        self.with_time = with_time
        self.__prepare_timestamps_mapping(list_of_files)
        self.__prepare_sequences(mode)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        data = []
        for timestamp in self.sequences[index]:
            with h5py.File(self.timestamp_to_file[timestamp]) as d:
                data.append(np.array(d[timestamp]["intensity"]))
        data = np.expand_dims(data, axis=1)
        data[data == -1e6] = 0
        data[data == -2e6] = -1
        inputs = data[: self.in_seq_len]
        targets = data[self.in_seq_len :]
        out = {"features": torch.from_numpy(inputs), "label": torch.from_numpy(targets)}
        if self.with_time:
            out["timestamp"] = torch.tensor(int(self.sequences[index][-1]))
        return out

    def __prepare_timestamps_mapping(self, list_of_files):
        self.timestamp_to_file = {}
        for filename in list_of_files:
            with h5py.File(filename) as d:
                self.timestamp_to_file = {
                    **self.timestamp_to_file,
                    **dict(map(lambda x: (x, filename), d.keys())),
                }

    def __prepare_sequences(self, mode):
        timestamps = np.unique(sorted(self.timestamp_to_file.keys()))
        if mode == "sequentially":
            self.sequences = [
                timestamps[index * self.seq_len : (index + 1) * self.seq_len]
                for index in range(len(timestamps) // self.seq_len)
            ]
        elif mode == "overlap":
            self.sequences = [
                timestamps[index : index + self.seq_len]
                for index in range(len(timestamps) - self.seq_len + 1)
            ]
        else:
            raise Exception(f"Unknown mode {mode}")
        self.sequences = list(
            filter(
                lambda x: int(x[-1]) - int(x[0]) == (self.seq_len - 1) * 600,
                self.sequences,
            )
        )


class Collator:
    def __init__(
        self,
        stage: str = "train",
        mix_proba: float = 0.5,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
    ) -> None:
        assert stage in ("train", "val", "test")
        self.stage = stage

        # mixup & cutmix augmentations
        self.mix_transform = None
        if (stage == "train") and (random.random() < mix_proba):
            mix_transforms = []
            if mixup_alpha > 0:
                mix_transforms.append(RandomMixup(alpha=mixup_alpha))
            if cutmix_alpha > 0:
                mix_transforms.append(RandomCutmix(alpha=cutmix_alpha))

            self.mix_transform = torchvision.transforms.RandomChoice(mix_transforms)

    def __call__(self, batch):
        seq_len, n_channels, height, width = batch[0]["features"].shape
        features = torch.zeros(len(batch), *batch[0]["features"].shape)

        has_label = "label" in batch[0]
        if has_label:
            labels = torch.zeros((len(batch), *batch[0]["label"].shape))
            # one_label = torch.zeros(len(batch))

        for idx, item in enumerate(batch):
            features[idx] = item["features"]
            if has_label:
                labels[idx] = item["label"]

        if has_label and self.mix_transform and (self.stage == "train"):
            features, labels = self.mix_transform(features, labels)

        out = {
            "features": features,
        }
        if has_label:
            out["label"] = labels

        return out
