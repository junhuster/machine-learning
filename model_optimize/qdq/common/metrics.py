#!/usr/bin/env python3
"""评估指标：只依赖 numpy，避免引入 sklearn 依赖。

AUC 用「秩和」公式计算（Mann-Whitney U），并对并列分数取平均秩：

    AUC = (sum(rank_pos) - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
"""
from __future__ import annotations

import numpy as np


def auc_score(y_true, y_score) -> float:
    """二分类 AUC。样本只有一个类别时返回 nan。"""
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(f"长度不一致: y_true={y_true.shape}, y_score={y_score.shape}")

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(y_score.shape[0], dtype=np.float64)

    n = y_score.shape[0]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # 名次从 1 开始，并列取平均
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = ranks[y_true == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def logloss(y_true, y_prob) -> float:
    """二分类交叉熵（输入为概率）。"""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), 1e-7, 1 - 1e-7)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


if __name__ == "__main__":
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    # 正类分数 0.35 / 0.8：0.35 只赢 0.1；0.8 赢 0.1,0.4  → 3 / 4 = 0.75
    print("auc =", auc_score(y, s), "(期望 0.75)")
