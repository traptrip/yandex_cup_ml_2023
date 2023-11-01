from pathlib import Path
from typing import Union

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        data_dir,
        num_labels=256,
        crop_size=60,
        stage="train",
        transform=None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.stage = stage
        self.transform = transform
        self.crop_size = crop_size
        self.num_labels = num_labels

        assert stage in ("train", "val", "test", "infer")

        if self.stage in ["train", "val", "test"]:
            self.meta_info = pd.read_csv(self.data_dir / "metadata.csv", sep="\t")
            if "stage" in self.meta_info.columns:
                self.meta_info = self.meta_info.loc[
                    self.meta_info.stage == stage
                ].reset_index(drop=True)
            self.labels = torch.tensor(
                self.meta_info.tags.apply(self.process_tags)
            ).float()
        else:
            self.meta_info = pd.read_csv(data_dir / "test.csv")

        self.tracks = self.meta_info.track.values

        print("Uploading embeddings into memory")
        self.embeddings = [self.get_embedding(track) for track in self.tracks]

    def process_tags(self, tags):
        tags = list(map(int, tags.split(",")))
        one_hot_tags = np.zeros(self.num_labels, dtype=np.uint8)
        one_hot_tags[tags] = 1
        return one_hot_tags.tolist()

    def get_embedding(self, track: int) -> torch.Tensor:
        embeddings = np.load(self.data_dir / f"track_embeddings/{track}.npy")
        embeddings = torch.from_numpy(embeddings)
        return embeddings

    def __process_features(self, x: torch.Tensor):
        # normalize
        # x /= x.max()
        # x = (x - x.mean()) / x.std()

        # add padding
        mask = torch.zeros(self.crop_size, dtype=torch.bool)
        x = x.permute(1, 0)
        x_len = x.shape[-1]
        if x_len > self.crop_size:
            start = np.random.randint(0, x_len - self.crop_size)
            x = x[..., start : start + self.crop_size]
        else:
            if self.stage == "train":
                left = (
                    np.random.randint(0, self.crop_size - x_len)
                    if self.crop_size != x_len
                    else 0
                )
            else:
                left = (self.crop_size - x_len) // 2
            right = self.crop_size - x_len - left
            pad_patern = (left, right)
            x = torch.nn.functional.pad(x, pad_patern, "constant").detach()
            mask[:left] = 1
            mask[left + x_len + right :] = 1
        x = x.permute(1, 0)
        return x, mask

    def __getitem__(self, idx):
        track_features = self.embeddings[idx]
        # track_features, mask = self.__process_features(track_features)
        out = {
            "features": track_features,
            # "mask": mask,
            "track": self.tracks[idx],
        }
        if self.stage != "infer":
            out["label"] = self.labels[idx]
        return out

    def __len__(self):
        return len(self.meta_info)


class Collator:
    def __init__(self, stage: str = "train") -> None:
        assert stage in ("train", "test")
        self.stage = stage

    def __call__(self, batch: list[dict[str, list[Union[torch.Tensor, int]]]]):
        if self.stage == "train":
            batch = sorted(
                batch, key=lambda value: len(value["features"]), reverse=True
            )
            max_len = len(batch[0]["features"])
        else:
            max_len = len(max(batch, key=lambda b: len(b["features"]))["features"])

        features = torch.zeros(len(batch), max_len, batch[0]["features"][0].shape[0])
        mask = torch.ones(len(batch), max_len, dtype=torch.bool)
        tracks = torch.zeros(len(batch), dtype=torch.long)
        has_label = "label" in batch[0]
        if has_label:
            labels = torch.zeros((len(batch), *batch[0]["label"].shape))

        for idx, item in enumerate(batch):
            x = item["features"]
            x_len = len(x)
            pad_left = (
                np.random.randint(0, max_len - x_len)
                if max_len != x_len and self.stage == "train"
                else 0
            )
            features[idx, pad_left : pad_left + x_len, :] = x
            mask[idx, pad_left : pad_left + x_len] = 0
            tracks[idx] = item["track"]

            if has_label:
                labels[idx] = item["label"]

        out = {
            "features": features,
            "mask": mask,
            "track": tracks,
        }
        if has_label:
            out["label"] = labels

        return out
