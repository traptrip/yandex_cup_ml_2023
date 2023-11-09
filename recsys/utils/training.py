import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score


def train(
    cfg,
    n_epochs,
    net,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    scheduler=None,
    label_smoothing=0.0,
    tb=None,  # tensorboard logger
    start_epoch=0,
):
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    alpha = 0.8
    best_val_score = 0

    for epoch in range(n_epochs):
        # Training
        net.train()
        train_loss = None
        train_targets = []
        train_preds = []
        for data in (pbar := tqdm(train_dataloader)):
            optimizer.zero_grad()
            for key in data:
                data[key] = data[key].to(cfg.device)
            track_ids, batch, mask, targets = (
                data["track"],
                data["features"],
                data["mask"],
                data["label"],
            )
            # one_label = data["one_label"]
            targets = targets * (1 - label_smoothing) + (label_smoothing / 2)

            with torch.cuda.amp.autocast(
                enabled=cfg.use_amp
            ), torch.backends.cuda.sdp_kernel(enable_flash=cfg.device == "cuda:0"):
                emb, logits = net(batch, mask)

                bce_loss = (
                    criterion["classification"]["w"]
                    * criterion["classification"]["f"](logits, targets)
                    if criterion["classification"]
                    else 0
                )
                emb_loss = (
                    criterion["embedding"]["w"]
                    * criterion["embedding"]["f"](emb, track_ids)
                    if criterion["embedding"]
                    else 0
                )
                loss = bce_loss + emb_loss

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
            train_targets.extend(targets.cpu().numpy() > 0.5)
            train_preds.extend(torch.sigmoid(logits.detach()).cpu().numpy())

            pbar.set_description(f"Epoch: {epoch} Loss: {train_loss:.6f}")

            del emb, logits, track_ids, batch, mask, targets

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
                    for key in data:
                        data[key] = data[key].to(cfg.device)
                    track_ids, batch, mask, targets = (
                        data["track"],
                        data["features"],
                        data["mask"],
                        data["label"],
                    )
                    # one_label = data["one_label"]

                    emb, logits = net(batch, mask)

                    bce_loss = (
                        criterion["classification"]["w"]
                        * criterion["classification"]["f"](logits, targets)
                        if criterion["classification"]
                        else 0
                    )
                    emb_loss = (
                        criterion["embedding"]["w"]
                        * criterion["embedding"]["f"](emb, track_ids)
                        if criterion["embedding"]
                        else 0
                    )
                    loss = bce_loss + emb_loss

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
        print("LR:", optimizer.param_groups[0]["lr"])
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
            tb.add_scalar("Loss/train", train_loss, epoch + start_epoch)
            tb.add_scalar("Loss/val", val_loss, epoch + start_epoch)
            tb.add_scalar("AP/train", train_score, epoch + start_epoch)
            tb.add_scalar("AP/val", val_score, epoch + start_epoch)
