from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import EmbeddingDataset, Collator
from utils.training import train
from utils.augmentations import RandomAugment


def run(
    cfg,
    n_epochs,
    net,
    optimizer,
    criterion,
    scheduler,
    max_crop,
    is_fixed_crop,
    start_epoch=0,
):
    tb = SummaryWriter(log_dir=cfg.logs_dir)

    train_dataset = EmbeddingDataset(
        cfg.data_dir,
        cfg.meta_info,
        cfg.num_labels,
        stage=cfg.train_stage,
        transform=RandomAugment(),
    )
    val_dataset = EmbeddingDataset(
        cfg.data_dir, cfg.meta_info, cfg.num_labels, stage="val"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=Collator(
            "train",
            cfg.num_labels,
            cfg.mix_proba,
            cfg.mixup_alpha,
            max_crop,
            is_fixed_crop,
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
            cfg.num_labels,
            cfg.mix_proba,
            cfg.mixup_alpha,
            max_crop,
            is_fixed_crop,
        ),
        drop_last=False,
    )

    net.to(cfg.device)

    train(
        cfg,
        n_epochs,
        net,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        scheduler,
        cfg.label_smoothing,
        tb,
        start_epoch=start_epoch,
    )

    del train_dataset, train_dataloader, val_dataset, val_dataloader


if __name__ == "__main__":
    run()
