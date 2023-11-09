import gc
import shutil
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import losses
import train
import inference
from utils.log_utils import get_exp_name
from optimizers.lion import Lion

# from models.mlp import Network

from models.transformer_encoder import Network


"""
TODO:
3. CNN1D
8. Emformer
0. gradient accumulation 
1. Autoencoder pretrain
7. MultiSimilarityLoss
3. NTXentLoss
9. SigLIP
2. SupAP pretraining (large batch size)

9. attention pooling
6. Увеличивать размер max_len по ходу обучения (curriculum learning)
4. Lion optimizer (можно взять больше батч)
10. Нормировать фичи
5. Train on Full dataset

"""
N_FOLDS = 10


class Config:
    # logging
    logs_dir: Path = get_exp_name(
        Path(
            f"_EXPERIMENTS/{N_FOLDS}folds__transformer__attn_pooling__flip_cut_aug__lion__resample"
        )
    )

    # data
    fold = None
    data_dir = Path("./data/")
    meta_info = Path("./data/metadata.csv")
    num_labels = 256
    train_stage = "train"

    batch_size = 96
    eval_batch_size = 96
    num_workers = 4

    # aug
    mix_proba = 1.0
    mixup_alpha = 1.0
    cutmix_alpha = 0.0  # TODO: bug?

    # net
    transformer_layers = 3
    num_heads = 8
    input_dim = 768
    hidden_dim = 512
    pooling = "attention"
    dim_feedforward = 2048
    dropout = 0.2

    device = "cuda:0"
    use_amp = True
    clip_value = 1
    lr = 3e-5  # lion: 3e-5  adamw: 1e-4
    min_lr = 1e-8

    max_crop = [None]
    is_fixed_crop = [False]
    n_epochs = [40]

    label_smoothing = 0.1

    cls_loss_args = {
        "resample_args": dict(
            use_sigmoid=True,
            reweight_func="rebalance",
            focal=dict(focal=True, balance_param=2.0, gamma=2),
            logit_reg=dict(neg_scale=2.0, init_bias=0.05),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
            loss_weight=1.0,
            device="cuda:0",
        )
    }

    emb_loss_args = {
        "weights": [1.0, 1.0],
        "calibration_args": {
            "pos_margin": 0.9,
            "neg_margin": 0.6,
        },
        "sup_ap_args": {
            "tau": 0.01,
            "rho": 100.0,
            "offset": 1.44,
            "delta": 0.05,
            "return_type": "1-mAP",
        },
    }


def log_current_state(cfg):
    cur_dir = Path(__file__).parent
    srcs = [
        cur_dir / "main.py",
        cur_dir / "folds_main.py",
        cur_dir / "train.py",
        cur_dir / "inference.py",
        cur_dir / "utils",
        cur_dir / "models",
    ]
    for src in srcs:
        if src.is_dir():
            shutil.copytree(src, cfg.logs_dir / src.name)
        else:
            shutil.copy(src, cfg.logs_dir / src.name)


if __name__ == "__main__":
    cfg = Config()
    print("Logging dir:", cfg.logs_dir)
    log_current_state(cfg)

    # create folds
    cfg.meta_info = pd.read_csv(cfg.data_dir / "train.csv")

    def one_hot_tags(tags):
        tags = list(map(int, tags.split(",")))
        one_hot = np.zeros(256)
        one_hot[tags] = 1
        return one_hot

    y = np.stack(cfg.meta_info.tags.apply(one_hot_tags))

    # kf = KFold(n_splits=N_FOLDS)
    kf = MultilabelStratifiedKFold(n_splits=N_FOLDS)
    folds = kf.split(cfg.meta_info, y)
    for n_fold, (train_index, test_index) in enumerate(folds):
        print(f"FOLD: {n_fold}")
        cfg.logs_dir = cfg.logs_dir / f"fold{n_fold}"
        cfg.fold = n_fold
        cfg.meta_info["stage"] = "train"
        cfg.meta_info.loc[test_index, "stage"] = "val"

        net = Network(
            transformer_layers=cfg.transformer_layers,
            num_heads=cfg.num_heads,
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_labels=cfg.num_labels,
            pooling=cfg.pooling,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )

        if isinstance(cfg.n_epochs, int):
            cfg.n_epochs //= len(cfg.max_crop)
            cfg.n_epochs = [cfg.n_epochs] * len(cfg.max_crop)

        for i, (max_crop, is_fixed_crop, n_epochs) in enumerate(
            zip(cfg.max_crop, cfg.is_fixed_crop, cfg.n_epochs)
        ):
            print(f"\n[{i}] MAX CROP: {max_crop}, IS FIXED: {is_fixed_crop}")

            criterion = {
                "classification": {
                    "w": 1.0,
                    "f": losses.ResampleLoss(**cfg.cls_loss_args["resample_args"]),
                },  # {"w": 1.0, "f": torch.nn.BCEWithLogitsLoss()},
                "embedding": None,
            }

            # optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
            optimizer = Lion(net.parameters(), lr=cfg.lr, weight_decay=0.01)

            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=n_epochs, eta_min=cfg.min_lr
                )
                if max_crop is None
                else None
            )
            train.run(
                cfg,
                n_epochs,
                net,
                optimizer,
                criterion,
                scheduler,
                max_crop,
                is_fixed_crop,
                start_epoch=sum(cfg.n_epochs[:i]),
            )

            del criterion, optimizer, scheduler
            torch.cuda.empty_cache()
            gc.collect()

        del net
        torch.cuda.empty_cache()
        gc.collect()
        cfg.logs_dir = cfg.logs_dir.parent

    # INFER
    print("\n\nINFERENCE")
    cfg.meta_info = Path("./data/test.csv")
    predictions = []
    for i in range(N_FOLDS):
        net = Network(
            transformer_layers=cfg.transformer_layers,
            num_heads=cfg.num_heads,
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_labels=cfg.num_labels,
            pooling=cfg.pooling,
        )
        net.load_state_dict(
            torch.load(
                cfg.logs_dir / f"fold{i}" / "weights/last.pt", map_location="cpu"
            )
        )
        track_idxs, pred = inference.run(
            cfg, net, cfg.max_crop[-1], cfg.is_fixed_crop[-1]
        )
        predictions.append(pred)

    predictions = np.mean(predictions, axis=0)

    print("Saving prediction")
    predictions_df = pd.DataFrame(
        [
            {"track": track, "prediction": ",".join([str(p) for p in probs])}
            for track, probs in zip(track_idxs, predictions)
        ]
    )
    predictions_df.to_csv(cfg.logs_dir / "prediction.csv", index=False)
