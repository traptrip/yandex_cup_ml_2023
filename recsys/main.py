import gc
import shutil
from pathlib import Path

import torch
import pandas as pd

import losses
import train
import inference
from utils.log_utils import get_exp_name
from optimizers.lion import Lion
from models.transformer_encoder import Network

# from models.mlp import Network
# from models.cnn1d import Network


"""
TODO:
8. Emformer
0. gradient accumulation 
1. Autoencoder pretrain
7. MultiSimilarityLoss
3. NTXentLoss
- RMSNorm в трансформере

- distribution postprocessing
- Hamming Loss
- Focal Loss
- weighted BCE с обучаемыми весами
- weighted BCE https://stackoverflow.com/questions/67730325/using-weights-in-crossentropyloss-and-bceloss-pytorch
- https://github.com/ShadeAlsha/LTR-weight-balancing/blob/master/demo2_second-stage-training.ipynb
- https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
- https://github.com/Vanint/Awesome-LongTailed-Learning/tree/main/Main-codebase/loss

- сделать undersampling, обучиться, потом сделать fine tune на всех данных
- бустинг из моделей (одна модель учится на ошибке предыдущей)
- 256 КЛАССИФИКАТОРОВ ПО ОДНОМУ НА КАЖДЫЙ КЛАСС :))))))
- gaussian blur на эмбеддинги :)


- прогнать фичи через преобразование фурье (а почему бы и нет)
- обучить две модели: одну на многочисленных таргетах, другую на хвосте
- ResampleLoss
- аугментация разворота фичей
- cutout
9. attention pooling
6. Увеличивать размер max_len по ходу обучения (curriculum learning)
4. Lion optimizer (можно взять больше батч)
10. Нормировать фичи
5. Train on Full dataset
3. CNN1D
9. SigLIP
2. SupAP pretraining (large batch size)
- увеличить weight decay

"""

SAMPLE = "tail50_"


class Config:
    # logging
    logs_dir: Path = get_exp_name(
        Path(
            f"_EXPERIMENTS/{SAMPLE}_transformer__attn_pooling__flip_cut_aug__lion__weighted_bce"
        )
    )

    # data
    fold = None
    data_dir = Path("./data/")
    meta_info = Path(f"./data/{SAMPLE}metadata.csv")
    num_labels = 206  # 256
    train_stage = "train"

    batch_size = 104  # 104 128
    eval_batch_size = 104
    num_workers = 4

    # aug
    mix_proba = 1.0
    mixup_alpha = 1.0
    cutmix_alpha = 0.0  # TODO: bug?

    # net
    transformer_layers = 3
    num_heads = 8
    input_dim = 768
    dim_feedforward = 2048
    dropout = 0.2
    hidden_dim = 512
    pooling = "attention"

    device = "cuda:1"
    use_amp = True
    clip_value = 1
    lr = 3e-5  # 3e-5  # lion: 3e-5  adamw: 1e-4
    min_lr = 1e-8

    # max_crop = [30, 80, None]
    # is_fixed_crop = [False, False, False]
    # n_epochs = [20, 10, 10]

    max_crop = [None]
    is_fixed_crop = [False]
    n_epochs = [35]

    label_smoothing = 0.1

    cls_loss_args = {
        "resample_args": dict(
            use_sigmoid=True,
            reweight_func="rebalance",
            focal=dict(focal=True, balance_param=2.0, gamma=2),
            logit_reg=dict(neg_scale=2.0, init_bias=0.05),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
            loss_weight=1.0,
            device="cuda:1",
        ),
        "weighted_bce": dict(
            device="cuda:1",
            weights=torch.load(f"./data/{SAMPLE}bce_class_weights.pth").to(device),
        ),
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
    try:
        print("Logging dir:", cfg.logs_dir)
        log_current_state(cfg)

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
                    "f": losses.WeightedBCEWithLogitsLoss(
                        **cfg.cls_loss_args["weighted_bce"]
                    ),
                    # "f": losses.ResampleLoss(**cfg.cls_loss_args["resample_args"]),
                },  # torch.nn.BCEWithLogitsLoss()},
                "embedding": None
                # "embedding": {
                #     "w": 0.0,
                #     "f": losses.CalibrationLoss(**cfg.emb_loss_args["calibration_args"]),
                # },
            }

            # optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
            optimizer = Lion(net.parameters(), lr=cfg.lr, weight_decay=0.01)

            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=n_epochs, eta_min=cfg.min_lr
                )
                if i == len(cfg.max_crop) - 1
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
    except KeyboardInterrupt:
        print("Stop training")

    # INFER
    print("INFERENCE")
    cfg.meta_info = Path("./data/test.csv")
    if isinstance(cfg.batch_size, list):
        cfg.batch_size = cfg.batch_size[-1]
    net.load_state_dict(
        torch.load(cfg.logs_dir / "weights/last.pt", map_location="cpu")
    )

    track_idxs, predictions = inference.run(
        cfg, net, cfg.max_crop[-1], cfg.is_fixed_crop[-1]
    )
    print("Saving prediction")
    predictions_df = pd.DataFrame(
        [
            {"track": track, "prediction": ",".join([str(p) for p in probs])}
            for track, probs in zip(track_idxs, predictions)
        ]
    )
    predictions_df.to_csv(cfg.logs_dir / "prediction.csv", index=False)
