#!/usr/bin/env python3
"""② 训练模型并保存 .pt。

用法:
    python3 train.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.conf import get_device, load_config, resolve, set_seed  # noqa: E402
from common.dataio import iter_batches, load_split  # noqa: E402
from common.metrics import auc_score  # noqa: E402
from model import build_model, count_params  # noqa: E402

INPUT_NAMES = ["user_id", "item_id", "cate_id", "history", "dense"]


@torch.no_grad()
def evaluate(model: nn.Module, bundle, device, batch_size: int = 4096) -> float:
    model.eval()
    scores, labels = [], []
    for batch in iter_batches(bundle, batch_size):
        b = batch.to(device)
        logits = model(**b.inputs(INPUT_NAMES))
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(b.label.cpu().numpy())
    return auc_score(np.concatenate(labels), np.concatenate(scores))


def main() -> int:
    cfg = load_config("config.yaml", __file__)
    t = cfg["train"]
    set_seed(t["seed"])
    device = get_device("cuda")

    data_path = resolve(cfg, cfg["data"]["data_path"])
    if not data_path.exists():
        raise SystemExit(f"找不到 {data_path}，请先跑 python3 gen_data.py")

    print(f"[train] 读取 {data_path.name} ...")
    train_set = load_split(data_path, "train")
    val_set = load_split(data_path, "val")
    print(f"[train] train={train_set.batch_size()} val={val_set.batch_size()} device={device}")

    model = build_model(cfg).to(device)
    print(f"[train] 参数量 {count_params(model) / 1e6:.2f} M")

    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    step = 0
    for epoch in range(t["epochs"]):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        seen = 0
        for batch in iter_batches(train_set, t["batch_size"], shuffle=True, seed=t["seed"] + epoch):
            b = batch.to(device)
            logits = model(**b.inputs(INPUT_NAMES))
            loss = loss_fn(logits, b.label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss) * b.batch_size()
            seen += b.batch_size()
            step += 1
            if step % t["log_every"] == 0:
                print(f"  epoch {epoch} step {step:5d} loss={float(loss):.4f}")

        val_auc = evaluate(model, val_set, device)
        print(f"[epoch {epoch}] loss={total_loss / max(seen, 1):.4f} "
              f"val_auc={val_auc:.4f} ({time.time() - t0:.1f}s)")

    out = resolve(cfg, t["save_path"])
    torch.save({"state_dict": model.state_dict(), "config": cfg}, out)
    print(f"[train] 已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
