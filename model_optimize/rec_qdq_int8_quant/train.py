"""
train.py — 训练模块（与 rec_pytorch_int8_quant 保持一致）
"""
import logging
import os
import torch
import torch.nn as nn
from evaluate import evaluate, avg_auc

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion  = nn.BCELoss()
    total_loss = 0.0
    for batch in loader:
        sparse    = batch["sparse"].to(device)
        dense     = batch["dense"].to(device)
        label_ctr = batch["label_ctr"].to(device)
        label_cvr = batch["label_cvr"].to(device)
        pred_ctr, pred_cvr = model(sparse, dense)
        loss = criterion(pred_ctr, label_ctr) + criterion(pred_cvr, label_cvr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train(model, train_loader, val_loader, cfg, device):
    t_cfg = cfg["train"]
    os.makedirs(os.path.dirname(t_cfg["checkpoint_path"]), exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=t_cfg["lr"],
                                 weight_decay=t_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_auc, best_metrics = 0.0, {}
    for epoch in range(1, t_cfg["epochs"] + 1):
        loss    = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        cur_auc = avg_auc(metrics)
        scheduler.step()
        logger.info("Epoch %d/%d | loss=%.4f | ctr_auc=%.4f | cvr_auc=%.4f | avg_auc=%.4f",
                    epoch, t_cfg["epochs"], loss,
                    metrics["ctr_auc"], metrics["cvr_auc"], cur_auc)
        if cur_auc > best_auc:
            best_auc     = cur_auc
            best_metrics = metrics
            torch.save(model.state_dict(), t_cfg["checkpoint_path"])
            logger.info("  >> New best saved (avg_auc=%.4f)", best_auc)

    logger.info("Training complete. Best avg_auc=%.4f", best_auc)
    return best_metrics
