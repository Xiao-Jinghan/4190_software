# core/models.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

from .fence_gan import FenceGAN
from .utils import make_windows_from_matrix, window_scores_to_time_series


# --------- 视角/预处理（示例）---------
class ResidualView:
    """按列做一阶差分残差视角"""
    def __init__(self, window: int = 32):
        self.window = window

    def __call__(self, X: np.ndarray) -> np.ndarray:
        T, C = X.shape
        out = np.zeros_like(X)
        out[1:] = X[1:] - X[:-1]
        return out


# --------- 基于 Fence-GAN 的检测器 ---------
class FenceGANDetector:
    """
    以窗口为样本训练 Fence-GAN（仅正常窗口），推断输出异常概率序列。
    """
    def __init__(
        self,
        window: int = 128,
        stride: int = 32,
        gamma: float = 0.5,
        lambda_disp: float = 1.0,
        z_dim: int = 64,
        device: str = "cuda",
    ):
        self.window = int(window)
        self.stride = int(stride)
        self.gamma = float(gamma)
        self.lambda_disp = float(lambda_disp)
        self.z_dim = int(z_dim)
        self.device = device
        self.model: Optional[FenceGAN] = None

    def fit(self, X: np.ndarray, labels: Optional[np.ndarray] = None, epochs: int = 50, batch_size: int = 128):
        X = np.asarray(X, dtype=np.float32)
        T, C = X.shape
        Xw = make_windows_from_matrix(X, self.window, self.stride)
        Nw = Xw.shape[0]
        Xw = Xw.reshape(Nw, -1)

        if labels is not None:
            lab_w = []
            for s in range(0, T - self.window + 1, self.stride):
                seg = labels[s: s + self.window]
                lab_w.append(int(np.all(seg == 0)))
            lab_w = np.asarray(lab_w)
            Xw_train = Xw[lab_w == 1]
        else:
            Xw_train = Xw

        self.model = FenceGAN(
            x_dim=Xw.shape[1],
            z_dim=self.z_dim,
            gamma=self.gamma,
            lambda_disp=self.lambda_disp,
            device=self.device,
        )
        self.model.fit(Xw_train, epochs=epochs, batch_size=batch_size)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "请先 fit()"
        T, C = X.shape
        Xw = make_windows_from_matrix(X, self.window, self.stride)
        Nw = Xw.shape[0]
        Xw = Xw.reshape(Nw, -1)
        w_scores = self.model.score(Xw)
        ts_scores = window_scores_to_time_series(w_scores, T, self.window, self.stride)
        return ts_scores
