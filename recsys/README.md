# Yandex Cup 2023 — ML: RecSys

https://contest.yandex.com/contest/54251/problems/ 

# Download data 
```
cd data
bash download_data.sh
```
Output: 
```text
data/
    download_data.sh
    track_embeddings.tar.gz
    data.tar.gz
    track_embeddings/
        0.npy
        ...
        76713.npy
    train.csv
    test.csv
```

# Prepare dataset for training
Run [utils/prepare_dataset.py](utils/prepare_dataset.py)
It will create 3 files in data directory: 
- `metadata.csv`           - dataset with `stage` column (train/val)
- `bce_class_weights.pth`  - tags weight for WeightedBCELoss
- `class_freq.pkl`         - tags frequency for ResampleLoss
```bash
python utils/prepare_dataset.py
```

# Run single experiment
1. Configure `Config` class in [main.py](./main.py)
2. Run training
```bash
python main.py
```

> All artifacts (logs, weights, files) will be saved in `Config.logs_dir`

# Run ensemble experiment
1. Configure `Config` class in [folds_main.py](./folds_main.py)
2. Run training
```bash
python folds_main.py
```

> All artifacts (logs, weights, files) will be saved in `Config.logs_dir`

# Tensorboard training/validation results analysis
```bash
tensorboard --logdir _EXPERIMENTS/
```
