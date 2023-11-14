import gc
import shutil
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

import losses
import train
import inference
from utils.log_utils import get_exp_name
from optimizers.lion import Lion
from utils.dataset import EmbeddingDataset, Collator
from torch.utils.data import DataLoader
from models.transformer_encoder import Network as TransformerNetwork
from models.transformer_encoder_rms import Network as TransformerRMSNetwork
from models.mlp import Network as MLPNetwork


N_FOLDS = 12


class Config:
    # logging
    logs_dir: Path = get_exp_name(Path(f"_EXPERIMENTS/{N_FOLDS}folds__transformer"))

    # data
    fold = None
    data_dir = Path("./data/")
    meta_info = Path("./data/metadata.csv")
    num_labels = 256

    # train_stage == train - use only train for training
    # train_stage == all - train on all data
    train_stage = "train"

    batch_size = 128
    eval_batch_size = 128
    num_workers = 4

    # aug
    mix_proba = 1.0
    mixup_alpha = 1.0

    # net
    net_name = "transformer"  # transformer transformer_rms mlp
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

    max_crop = [120]
    is_fixed_crop = [False]
    n_epochs = [40]

    maxperfold = 0  # 4 #  use only for inference

    label_smoothing = 0.05

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
        cur_dir / "losses",
        cur_dir / "optimizers",
    ]
    for src in srcs:
        if src.is_dir():
            shutil.copytree(src, cfg.logs_dir / src.name)
        else:
            shutil.copy(src, cfg.logs_dir / src.name)


def get_network(cfg: Config):
    if cfg.net_name == "transformer":
        net = TransformerNetwork(
            transformer_layers=cfg.transformer_layers,
            num_heads=cfg.num_heads,
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_labels=cfg.num_labels,
            pooling=cfg.pooling,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )
    elif cfg.net_name == "transformer_rms":
        net = TransformerRMSNetwork(
            transformer_layers=cfg.transformer_layers,
            num_heads=cfg.num_heads,
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_labels=cfg.num_labels,
            pooling=cfg.pooling,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )
    elif cfg.net_name == "mlp":
        net = MLPNetwork(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_labels=cfg.num_labels,
        )
    else:
        raise ValueError(f"Network {cfg.net_name} is not implemented!")

    return net


if __name__ == "__main__":
    cfg = Config()
    print("Logging dir:", cfg.logs_dir)
    log_current_state(cfg)

    # Create folds
    cfg.meta_info = pd.read_csv(cfg.data_dir / "train.csv")

    def one_hot_tags(tags):
        tags = list(map(int, tags.split(",")))
        one_hot = np.zeros(256)
        one_hot[tags] = 1
        return one_hot

    y = np.stack(cfg.meta_info.tags.apply(one_hot_tags))
    kf = KFold(n_splits=N_FOLDS)
    folds = list(kf.split(cfg.meta_info, y))

    # Train folds
    for n_fold, (train_index, test_index) in enumerate(folds):
        print(f"FOLD: {n_fold}")
        cfg.logs_dir = cfg.logs_dir / f"fold{n_fold}"
        cfg.fold = n_fold
        cfg.meta_info["stage"] = "train"
        cfg.meta_info.loc[test_index, "stage"] = "val"

        net = get_network(cfg)

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
                    # "f": torch.nn.BCEWithLogitsLoss(),
                },
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

    # Inference
    print("\n\nINFERENCE")
    cfg.meta_info = Path("./data/test.csv")
    predictions = []

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
            max_crop_size=cfg.max_crop[-1],
            is_fixed_crop=cfg.is_fixed_crop[-1],
        ),
    )
    for i in range(N_FOLDS):
        print(f"Fold #{i}")
        net = get_network(cfg)
        net.load_state_dict(
            torch.load(
                cfg.logs_dir / f"fold{i}" / "weights/best.pt", map_location="cpu"
            )
        )

        if cfg.maxperfold > 0:
            preds = []
            for _ in range(cfg.maxperfold):
                track_idxs, pred = inference.run(
                    cfg, net, cfg.max_crop[-1], cfg.is_fixed_crop[-1], test_dataloader
                )
                preds.append(pred)
            preds = np.max(preds, axis=0)
        else:
            track_idxs, preds = inference.run(
                cfg, net, cfg.max_crop[-1], cfg.is_fixed_crop[-1], test_dataloader
            )

        predictions.append(preds)

    predictions = np.mean(predictions, axis=0)

    print("Saving prediction")
    predictions_df = pd.DataFrame(
        [
            {"track": track, "prediction": ",".join([str(p) for p in probs])}
            for track, probs in zip(track_idxs, predictions)
        ]
    )
    predictions_df.to_csv(cfg.logs_dir / f"prediction.csv", index=False)
