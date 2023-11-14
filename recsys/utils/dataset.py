import random
import json
from pathlib import Path

import torch
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn import preprocessing

from .augmentations import RandomCutmix, RandomMixup


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        data_dir,
        meta_info: str | Path | pd.DataFrame,
        num_labels=256,
        stage="train",
        transform=None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.stage = stage
        self.transform = transform
        self.num_labels = num_labels
        self.id_encoder = preprocessing.LabelEncoder()

        assert stage in ("all", "train", "val", "test", "infer")

        if isinstance(meta_info, (str, Path)):
            self.meta_info = pd.read_csv(meta_info)
        elif isinstance(meta_info, pd.DataFrame):
            self.meta_info = meta_info.copy()

        if self.stage in ["all", "train", "val", "test"]:
            # Encode track ids
            self.meta_info["track"] = self.id_encoder.fit_transform(
                self.meta_info["track"]
            )
            self.trackid2track = {
                i: cls for i, cls in enumerate(self.id_encoder.classes_.tolist())
            }
            # self.trackid2track = {cls: cls for cls in self.meta_info.track}
            with open(self.data_dir / f"trackid2track.txt", "w") as f:
                json.dump(self.trackid2track, f)

            if (self.stage != "all") and ("stage" in self.meta_info.columns):
                self.meta_info = self.meta_info.loc[
                    self.meta_info.stage == stage
                ].reset_index(drop=True)
            self.labels = torch.tensor(
                self.meta_info.tags.apply(self.process_tags)
            ).float()
            # self.one_label = torch.from_numpy(self.meta_info.one_label.values)
        else:
            # testing
            # self.meta_info = pd.read_csv(data_dir / "test.csv")
            self.trackid2track = {cls: cls for cls in self.meta_info.track}

        self.tracks = self.meta_info.track.values

        print("Uploading embeddings into memory")
        self.embeddings = {
            track: self.get_embedding(self.trackid2track[track])
            for track in set(self.tracks)
        }

    def process_tags(self, tags):
        tags = list(map(int, tags.split(",")))
        one_hot_tags = np.zeros(self.num_labels, dtype=np.uint8)
        one_hot_tags[tags] = 1
        return one_hot_tags.tolist()

    def get_embedding(self, track: int) -> torch.Tensor:
        embeddings = np.load(self.data_dir / f"track_embeddings/{track}.npy")
        embeddings = torch.from_numpy(embeddings)
        return embeddings

    def __getitem__(self, idx):
        track = self.tracks[idx]
        track_features = self.embeddings[track]
        # track_features = torch.sin(track_features) + torch.cos(track_features)

        if self.transform:
            track_features = self.transform(track_features)

        out = {
            "features": track_features,
            "track": track,
        }
        if self.stage != "infer":
            out["label"] = self.labels[idx]
            # out["one_label"] = self.one_label[idx]
        return out

    def __len__(self):
        return len(self.meta_info)


class Collator:
    def __init__(
        self,
        stage: str = "train",
        num_labels: int = 256,
        mix_proba: float = 1.0,
        mixup_alpha: float = 1.0,
        max_crop_size: int = None,
        is_fixed_crop: bool = False,
    ) -> None:
        assert stage in ("train", "val", "test")
        self.stage = stage
        self.max_crop_size = max_crop_size
        self.is_fixed_crop = is_fixed_crop

        # mixup augmentations
        self.mix_transform = None
        if (stage == "train") and (random.random() < mix_proba):
            if mixup_alpha > 0:
                self.mix_transform = RandomMixup(num_labels, alpha=mixup_alpha)

    def __call__(self, batch):
        # sort by seq len
        if self.stage == "train":
            batch = sorted(
                batch, key=lambda value: len(value["features"]), reverse=True
            )
            max_len = len(batch[0]["features"])
        else:
            max_len = len(max(batch, key=lambda b: len(b["features"]))["features"])

        # crop if max_crop_size specified
        if self.max_crop_size:
            max_len = (
                self.max_crop_size
                if self.is_fixed_crop
                else min(self.max_crop_size, max_len)
            )

        features = torch.zeros(len(batch), max_len, batch[0]["features"][0].shape[0])
        mask = torch.ones(len(batch), max_len, dtype=torch.bool)
        tracks = torch.zeros(len(batch), dtype=torch.long)

        has_label = "label" in batch[0]
        if has_label:
            labels = torch.zeros((len(batch), *batch[0]["label"].shape))

        for idx, item in enumerate(batch):
            x = item["features"]
            x_len = len(x)

            # cut mix
            do_cutmix = random.random() < 0.5 and self.stage == "train"
            if do_cutmix:
                idx_rolled = (idx + 1) % len(batch)
                x_rolled = batch[idx_rolled]["features"]
                x_rolled_len = len(x_rolled)

                seq_len = min(x_len, x_rolled_len)

                crop_part = int(seq_len * random.random() * 0.5)
                start1 = random.randint(0, x_len - crop_part)
                end1 = start1 + crop_part

                start2 = random.randint(0, x_rolled_len - crop_part)
                end2 = start2 + crop_part

                x[start1:end1] = x_rolled[start2:end2]

            if x_len > max_len:
                start = (
                    np.random.randint(0, x_len - max_len)
                    if self.stage == "train"
                    else 0
                )
                x = x[start : start + max_len]
                x_len = max_len

            # padding
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
                if do_cutmix:
                    labels[idx] = torch.logical_or(
                        labels[idx], batch[idx_rolled]["label"]
                    )

        if (
            has_label
            and not do_cutmix
            and self.mix_transform
            and (self.stage == "train")
        ):
            features, labels, mask = self.mix_transform(features, labels, mask)

        out = {
            "features": features,
            "mask": mask,
            "track": tracks,
        }
        if has_label:
            out["label"] = labels

        return out
