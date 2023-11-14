import pickle
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "../data"
DEFAULT_TEST_SIZE = 0.1
SEED = 42


def prepare_metadata_file(data: pd.DataFrame, test_size=DEFAULT_TEST_SIZE):
    train, val = train_test_split(data, test_size=test_size, random_state=SEED)
    train["stage"] = "train"
    val["stage"] = "val"
    meta = pd.concat((train, val), ignore_index=True)
    meta.to_csv(DATA_DIR / "metadata.csv", index=False)


def process_tags(tags):
    tags = list(map(int, tags.split(",")))
    return tags


def prepare_weights_for_wbce(data: pd.DataFrame):
    tags = data.tags.apply(process_tags)
    all_tags = tags.explode()
    tags_count = all_tags.value_counts()
    all_tags = all_tags.values.astype(np.int32)

    weights = torch.from_numpy(1 / tags_count.sort_index().values).float()
    torch.save(weights, DATA_DIR / "bce_class_weights.pth")


def prepare_weights_for_resample_loss(data: pd.DataFrame):
    tags = data.tags.apply(process_tags)
    all_tags = tags.explode()
    tags_cnt = all_tags.value_counts().sort_index()

    class_freq = tags_cnt.values
    neg_class_freq = len(data) - class_freq

    res = {
        "class_freq": class_freq,
        "neg_class_freq": neg_class_freq,
    }

    with open(DATA_DIR / "class_freq.pkl", "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    prepare_metadata_file(train_df)
    prepare_weights_for_wbce(train_df)
    prepare_weights_for_resample_loss(train_df)
