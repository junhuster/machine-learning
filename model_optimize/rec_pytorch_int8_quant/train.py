"""
train.py — 训练模块
训练 MMOE 模型并保存最优检查点
"""

import logging
import os
import torch
import torch.nn as nn
from evaluate import evaluate, avg_auc

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, device):
    """跑一个 epoch，返回平均 loss。"""
    model.train()
    criterion = nn.BCELoss()
    total_loss = 0.0
    for batch in loader:
        sparse     = batch["sparse"].to(device)
        dense      = batch["dense"].to(device)
        label_ctr  = batch["label_ctr"].to(device)
        label_cvr  = batch["label_cvr"].to(device)

        pred_ctr, pred_cvr = model(sparse, dense)
        loss = criterion(pred_ctr, label_ctr) + criterion(pred_cvr, label_cvr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train(model, train_loader, val_loader, cfg, device):
    """
    完整训练流程，每个 epoch 在验证集评估，保存最优模型。

    Returns:
        best_metrics (dict): 最优 epoch 的 ctr_auc / cvr_auc
    """
    train_cfg = cfg["train"]
    os.makedirs(os.path.dirname(train_cfg["checkpoint_path"]), exist_ok=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_auc = 0.0
    best_metrics = {}

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics    = evaluate(model, val_loader, device)
        current_auc = avg_auc(metrics)
        scheduler.step()

        logger.info(
            "Epoch %d/%d | loss=%.4f | ctr_auc=%.4f | cvr_auc=%.4f | avg_auc=%.4f",
            epoch, train_cfg["epochs"],
            train_loss, metrics["ctr_auc"], metrics["cvr_auc"], current_auc,
        )

        if current_auc > best_auc:
            best_auc = current_auc
            best_metrics = metrics
            torch.save(model.state_dict(), train_cfg["checkpoint_path"])
            logger.info("  >> New best model saved (avg_auc=%.4f)", best_auc)

    logger.info("Training complete. Best avg_auc=%.4f", best_auc)
    return best_metrics
