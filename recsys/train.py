import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.log_utils import get_exp_name
from utils.dataset import EmbeddingDataset, Collator
from utils.training import train

# from models.transformer_encoder import Network
from models.mlp import Network

"""
TODO:
1. добавить mixup + cutmix аугментации 
4. добавить supAP лосс
"""


class Config:
    # logging
    logs_dir: Path = get_exp_name(Path("_EXPERIMENTS/mlp"))

    # data
    data_dir = Path("./data/")
    num_labels = 256
    crop_size = None

    batch_size = 1024  # 640 is use_amp
    eval_batch_size = 1024
    num_workers = 0

    # net
    transformer_layers = 1
    num_heads = 8
    input_dim = 768
    hidden_dim = 512

    device = "cuda:1"
    use_amp = False
    clip_value = None
    lr = 3e-4
    min_lr = 1e-8

    n_epochs = 80


def log_current_state(cfg):
    cur_dir = Path(__file__).parent
    srcs = [
        cur_dir / "train.py",
        cur_dir / "utils",
        cur_dir / "models",
    ]
    for src in srcs:
        if src.is_dir():
            shutil.copytree(src, cfg.logs_dir / src.name)
        else:
            shutil.copy(src, cfg.logs_dir / src.name)


def main():
    cfg = Config()

    print("Logging dir:", cfg.logs_dir)
    log_current_state(cfg)
    tb = SummaryWriter(log_dir=cfg.logs_dir)

    train_dataset = EmbeddingDataset(
        cfg.data_dir, cfg.num_labels, cfg.crop_size, stage="train"
    )
    val_dataset = EmbeddingDataset(
        cfg.data_dir, cfg.num_labels, cfg.crop_size, stage="val"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=Collator(),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=Collator(),
    )

    # net = Network(
    #     transformer_layers=cfg.transformer_layers,
    #     num_heads=cfg.num_heads,
    #     input_dim=cfg.input_dim,
    #     hidden_dim=cfg.hidden_dim,
    #     num_labels=cfg.num_labels,
    # ).to(cfg.device)
    net = Network(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_labels=cfg.num_labels,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=cfg.min_lr
    )

    train(
        cfg, net, train_dataloader, val_dataloader, optimizer, criterion, scheduler, tb
    )


if __name__ == "__main__":
    main()
