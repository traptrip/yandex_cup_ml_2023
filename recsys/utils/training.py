import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score


def train(
    cfg,
    net,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    scheduler=None,
    tb=None,  # tensorboard logger
):
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    alpha = 0.8
    best_val_score = 0

    for epoch in range(cfg.n_epochs):
        # Training
        net.train()
        train_loss = None
        train_targets = []
        train_preds = []
        for data in (pbar := tqdm(train_dataloader)):
            optimizer.zero_grad()
            track_ids, batch, mask, targets = (
                data["track"],
                data["features"],
                data["mask"],
                data["label"],
            )
            batch = batch.to(cfg.device)
            mask = mask.to(cfg.device)
            targets = targets.to(cfg.device)
            with torch.cuda.amp.autocast(
                enabled=cfg.use_amp
            ), torch.backends.cuda.sdp_kernel(enable_flash=cfg.device == "cuda:0"):
                logits = net(batch, mask)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if cfg.clip_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_value)
            scaler.step(optimizer)
            scaler.update()

            train_loss = (
                loss.item()
                if not train_loss
                else alpha * train_loss + (1 - alpha) * loss.item()
            )
            train_targets.extend(targets.cpu().numpy())
            train_preds.extend(torch.sigmoid(logits.detach()).cpu().numpy())

            pbar.set_description(f"Epoch: {epoch} Loss: {train_loss:.6f}")

        if scheduler:
            scheduler.step()

        train_loss = np.mean(train_loss)
        train_score = average_precision_score(train_targets, train_preds)
        print("Train Loss:", train_loss)
        print("Train AP:", train_score)

        # Evaluation
        net.eval()
        val_loss = None
        val_targets = []
        val_preds = []
        for data in (pbar := tqdm(val_dataloader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=cfg.use_amp
                ), torch.backends.cuda.sdp_kernel(enable_flash=cfg.device == "cuda:0"):
                    track_ids, batch, mask, targets = (
                        data["track"],
                        data["features"],
                        data["mask"],
                        data["label"],
                    )
                    # track_ids, batch, targets = data
                    batch = batch.to(cfg.device)
                    mask = mask.to(cfg.device)
                    targets = targets.to(cfg.device)

                    logits = net(batch, mask)
                    loss = criterion(logits, targets.float())

                val_loss = (
                    loss.item()
                    if not val_loss
                    else alpha * val_loss + (1 - alpha) * loss.item()
                )
                val_targets.extend(targets.cpu().numpy())
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())

                pbar.set_description(f"Epoch: {epoch} Loss: {val_loss:.6f}")

        val_loss = np.mean(val_loss)
        val_score = average_precision_score(val_targets, val_preds)
        print("Val Loss:", val_loss)
        print("Val AP:", val_score)

        (cfg.logs_dir / "weights").mkdir(exist_ok=True)
        torch.save(net.state_dict(), cfg.logs_dir / "weights" / "last.pt")
        if val_score > best_val_score:
            print(f"Score improved from {best_val_score:.4f} to {val_score:.4f}")
            best_val_score = val_score
            torch.save(net.state_dict(), cfg.logs_dir / "weights" / "best.pt")
            with open(cfg.logs_dir / "weights" / "best_score.txt", "w") as f:
                f.write(f"Train AP: {train_score}\nVal AP: {val_score}")
        print()

        if tb is not None:
            tb.add_scalar("Loss/train", train_loss, epoch)
            tb.add_scalar("Loss/val", val_loss, epoch)
            tb.add_scalar("AP/train", train_score, epoch)
            tb.add_scalar("AP/val", val_score, epoch)
