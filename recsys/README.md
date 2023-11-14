# Yandex Cup 2023 - ML: RecSys
> Python 3.10.12

**Ledearboard**: 3rd place \
**Public score**: 0.3066	\
**Private score**: 0.3110

[Task description](https://contest.yandex.com/contest/54251/problems/ ): \
Predict a music tags probabilities. Tags quantity is `256`.
Each track is described by embeddings of dimension (N, 768), where N depends on the track length.
The main testing metric is [Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

<img width="618" alt="image" src="https://github.com/traptrip/yandex_cup_ml_2023/blob/main/recsys/assets/leaderboard.png">



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
2. Use your own `optimizer`, `criterion`, `scheduler` if needed. You can change it in main function
3. Run training
```bash
python main.py
```

> All artifacts (logs, weights, files) will be saved in `Config.logs_dir`

# Run ensemble experiment
1. Configure `Config` class in [folds_main.py](./folds_main.py)
2. Use your own `optimizer`, `criterion`, `scheduler` if needed. You can change it in main function
3. Run training
```bash
python folds_main.py
```

> All artifacts (logs, weights, files) will be saved in `Config.logs_dir`

# Tensorboard training/validation results analysis
```bash
tensorboard --logdir _EXPERIMENTS/
```
