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
    label_smoothing=0.0,
    tb=None,  # tensorboard logger
):
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    alpha = 0.8
    best_val_score = 0

    for epoch in range(cfg.n_epochs):
        # Training
        net.train()
        train_loss = None
        train_score = np.zeros((12,), dtype=float)
        for batch_idx, data in enumerate(pbar := tqdm(train_dataloader)):
            # if batch_idx > 50:
            #     break
            optimizer.zero_grad()
            for key in data:
                data[key] = data[key].to(cfg.device)
            batch, targets = data["features"], data["label"]
            mask = data["mask"] if "mask" in data else None
            targets = targets * (1 - label_smoothing) + (label_smoothing / 2)

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                logits = net(batch, mask)
                logits[targets == -1] = -1
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
            train_score += np.sum(
                (
                    np.square(
                        targets.detach().cpu().numpy() - logits.detach().cpu().numpy()
                    )
                )
                * (targets.detach().cpu().numpy() != -1),
                axis=(0, 2, 3, 4),
            )

            pbar.set_description(
                f"Epoch: {epoch} Loss: {train_loss:.6f} Score: {np.mean(np.sqrt(train_score / (batch_idx + 1))):.2f}"
            )

            del logits, batch, mask, targets

        if scheduler:
            scheduler.step()

        train_score /= batch_idx + 1
        train_score = np.mean(np.sqrt(train_score))
        print("Train Loss:", train_loss)
        print("Train score:", train_score)

        # Evaluation
        net.eval()
        val_loss = None
        val_score = np.zeros((12,), dtype=float)
        for batch_idx, data in enumerate(pbar := tqdm(val_dataloader)):
            # if batch_idx > 50:
            #     break
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    for key in data:
                        data[key] = data[key].to(cfg.device)
                    batch, targets = data["features"], data["label"]
                    mask = data["mask"] if "mask" in data else None

                    logits = net(batch, mask)
                    logits[targets == -1] = -1
                    loss = criterion(logits, targets)

                val_loss = (
                    loss.item()
                    if not val_loss
                    else alpha * val_loss + (1 - alpha) * loss.item()
                )
                val_score += np.sum(
                    (
                        np.square(
                            targets.detach().cpu().numpy()
                            - logits.detach().cpu().numpy()
                        )
                    )
                    * (targets.detach().cpu().numpy() != -1),
                    axis=(0, 2, 3, 4),
                )

                pbar.set_description(
                    f"Epoch: {epoch} Loss: {val_loss:.6f} Score: {np.mean(np.sqrt(val_score / (batch_idx + 1))):.2f}"
                )

        val_score /= batch_idx + 1
        val_score = np.mean(np.sqrt(val_score))
        print("LR:", optimizer.param_groups[0]["lr"])
        print("Val Loss:", val_loss)
        print("Val score:", val_score)

        (cfg.logs_dir / "weights").mkdir(exist_ok=True)
        torch.save(net.state_dict(), cfg.logs_dir / "weights" / "last.pt")
        if val_score > best_val_score:
            print(f"Score improved from {best_val_score:.4f} to {val_score:.4f}")
            best_val_score = val_score
            torch.save(net.state_dict(), cfg.logs_dir / "weights" / "best.pt")
        with open(cfg.logs_dir / "weights" / "best_score.txt", "w") as f:
            f.write(
                f"Epoch: {epoch} \nTrain AP: {train_score} \nVal AP: {val_score} \nBest val score: {best_val_score}"
            )
        print()

        if tb is not None:
            tb.add_scalar("Loss/train", train_loss, epoch)
            tb.add_scalar("Loss/val", val_loss, epoch)
            tb.add_scalar("Score/train", train_score, epoch)
            tb.add_scalar("Score/val", val_score, epoch)
