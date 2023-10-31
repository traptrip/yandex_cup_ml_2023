from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.log_utils import get_exp_name
from utils.dataset import EmbeddingDataset, Collator
from models.transformer_encoder import Network
from utils.training import train


"""
TODO:
1. добавить mixup + cutmix аугментации 
2. добавить random start_idx в Collator
3. добавить более умную инициализацию весов
4. добавить supAP лосс
"""
class Config:
    # logging
    logs_dir: Path = get_exp_name(Path("_EXPERIMENTS/train"))

    # data
    data_dir = Path("./data/")
    num_labels = 256
    crop_size = None

    batch_size = 160  # 640 is use_amp
    eval_batch_size = 160
    num_workers = 4

    # net
    input_dim = 768
    hidden_dim = 512

    device = "cuda:1"
    use_amp = False
    clip_value = None
    lr = 3e-4
    min_lr = 1e-8

    n_epochs = 80


def main():
    cfg = Config()

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

    net = Network(
        transformer_layers=1,
        num_heads=8,
        input_dim=768,
        hidden_dim=512,
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
