import pickle
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "../../data/"


def prepare(data: pd.DataFrame):
    def process_tags(tags):
        tags = list(map(int, tags.split(",")))
        return tags

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
    data = pd.read_csv(DATA_DIR / "train.csv")
    prepare(data)
