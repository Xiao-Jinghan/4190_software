# core/utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd


# -----------------------------
# 滑窗构造
# -----------------------------
def make_windows_from_df(df: pd.DataFrame, window: int, stride: int) -> np.ndarray:
    X = df.values.astype(np.float32)
    return make_windows_from_matrix(X, window, stride)


def make_windows_from_matrix(X: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    X: (T, C) -> (Nw, W, C)
    """
    X = np.asarray(X, dtype=np.float32)
    T, C = X.shape
    arr = []
    for i in range(0, T - window + 1, stride):
        arr.append(X[i:i + window, :])
    if not arr:
        return np.zeros((0, window, C), dtype=np.float32)
    return np.stack(arr, axis=0)


# -----------------------------
# 将窗口分数映射回时间序列
# -----------------------------
def window_scores_to_time_series(w_scores: np.ndarray, T: int, window: int, stride: int) -> np.ndarray:
    """
    w_scores: (Nw,)  每个窗口的异常分数（越大越异常，0~1）
    返回: (T,)   逐时刻分数（窗口覆盖平均）
    """
    w_scores = np.asarray(w_scores, dtype=np.float32).reshape(-1)
    ts = np.zeros(T, dtype=np.float32)
    ct = np.zeros(T, dtype=np.float32)

    idx = 0
    for start in range(0, T - window + 1, stride):
        ts[start:start + window] += w_scores[idx]
        ct[start:start + window] += 1.0
        idx += 1

    ct[ct == 0] = 1.0
    ts = ts / ct

    # 对未覆盖到的尾部做延伸（一般 stride < window 时不会发生）
    last = float(ts[0]) if T > 0 else 0.0
    for i in range(T):
        if ct[i] == 0:
            ts[i] = last
        else:
            last = ts[i]

    return np.clip(ts, 0.0, 1.0)
