from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import RadarDataset, Collator
from utils.training import train
from utils.augmentations import RandomAugment


def run(cfg, net, optimizer, criterion, scheduler):
    tb = SummaryWriter(log_dir=cfg.logs_dir)

    train_dataset = RadarDataset(cfg.train_files, mode=cfg.mode)
    val_dataset = RadarDataset(cfg.valid_files, mode=cfg.mode)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=Collator(
            "train",
            cfg.mix_proba,
            cfg.mixup_alpha,
            cfg.cutmix_alpha,
        ),
        drop_last=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=Collator(
            "val",
            cfg.mix_proba,
            cfg.mixup_alpha,
            cfg.cutmix_alpha,
        ),
        drop_last=False,
    )

    net.to(cfg.device)

    train(
        cfg,
        net,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        scheduler,
        cfg.label_smoothing,
        tb,
    )

    del train_dataset, train_dataloader, val_dataset, val_dataloader


if __name__ == "__main__":
    run()
