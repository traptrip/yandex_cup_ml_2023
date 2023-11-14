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
from models.convlstm import Seq2Seq
from models.ae import AutoencoderKL, SimpleConvDecoder, SimpleConvEncoder


class Config:
    # logging
    logs_dir: Path = get_exp_name(Path(f"_EXPERIMENTS/all_data_ae"))

    # data
    data_dir = Path("./data/")
    train_files = [
        "./data/ML Cup 2023 Weather/train/2021-01-train.hdf5",
        "./data/ML Cup 2023 Weather/train/2021-02-train.hdf5",
        "./data/ML Cup 2023 Weather/train/2021-03-train.hdf5",
        "./data/ML Cup 2023 Weather/train/2021-04-train.hdf5",
        "./data/ML Cup 2023 Weather/train/2021-05-train.hdf5",
        "./data/ML Cup 2023 Weather/train/2021-06-train.hdf5",
        "./data/ML Cup 2023 Weather/train/2021-07-train.hdf5",
        "./data/ML Cup 2023 Weather/train/2021-08-train.hdf5",
        "./data/ML Cup 2023 Weather/train/2021-09-train.hdf5",
        "./data/ML Cup 2023 Weather/train/2021-10-train.hdf5",
        "./data/ML Cup 2023 Weather/train/2021-11-train.hdf5",
        "./data/ML Cup 2023 Weather/train/2021-12-train.hdf5",
    ]
    valid_files = [
        # "./data/ML Cup 2023 Weather/train/2021-02-train.hdf5",
        # "./data/ML Cup 2023 Weather/train/2021-05-train.hdf5",
        # "./data/ML Cup 2023 Weather/train/2021-08-train.hdf5",
        "./data/ML Cup 2023 Weather/train/2021-11-train.hdf5",
    ]
    test_files = ["./data/ML Cup 2023 Weather/2022-test-public.hdf5"]
    mode = "overlap"

    batch_size = 1  # 104 128
    eval_batch_size = 1
    num_workers = 12

    # aug
    mix_proba = 1.0
    mixup_alpha = 1.0
    cutmix_alpha = 1.0

    # net
    transformer_layers = 1
    num_heads = 1
    input_dim = 768
    dim_feedforward = 2048
    dropout = 0.2
    hidden_dim = 512
    pooling = "attention"

    device = "cuda:0"
    use_amp = True
    clip_value = 1
    lr = 3e-5  # 3e-5  # lion: 3e-5  adamw: 1e-4
    min_lr = 1e-8

    n_epochs = 3

    label_smoothing = 0.0


def log_current_state(cfg):
    cur_dir = Path(__file__).parent
    srcs = [
        cur_dir / "main.py",
        cur_dir / "train.py",
        cur_dir / "inference.py",
        cur_dir / "utils",
        cur_dir / "models",
        cur_dir / "optimizers",
        cur_dir / "losses",
    ]
    for src in srcs:
        if src.is_dir():
            shutil.copytree(src, cfg.logs_dir / src.name)
        else:
            shutil.copy(src, cfg.logs_dir / src.name)


if __name__ == "__main__":
    cfg = Config()

    net = Seq2Seq(
        num_channels=1,
        num_kernels=32,
        kernel_size=(3, 3),
        padding=(1, 1),
        activation="relu",
        num_layers=1,
        out_seq_len=12,
    )
    # net = AutoencoderKL(SimpleConvEncoder(), SimpleConvDecoder())

    try:
        print("Logging dir:", cfg.logs_dir)
        log_current_state(cfg)

        criterion = torch.nn.MSELoss()

        # optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
        optimizer = Lion(net.parameters(), lr=cfg.lr, weight_decay=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.n_epochs, eta_min=cfg.min_lr
        )
        train.run(
            cfg,
            net,
            optimizer,
            criterion,
            scheduler,
        )

        del criterion, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()
    except KeyboardInterrupt:
        print("Stop training")

    # INFER
    print("INFERENCE")
    net.load_state_dict(
        torch.load(cfg.logs_dir / "weights/last.pt", map_location="cpu")
    )

    inference.run(cfg, net)
