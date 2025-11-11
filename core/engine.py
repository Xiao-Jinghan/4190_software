# core/engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from .fence_gan import FenceGAN
from .utils import (
    make_windows_from_matrix,
    window_scores_to_time_series,
)


class MimicDefenseEngine:
    """
    拟态防御核心引擎：
      - 支持前端动态传入 nodes / edges / fencegan_cfg
      - fencegan_cfg 可兼容旧版字段（latent_dim、boundary_gamma、dispersion_lambda 等）
      - detect() 为统一外部接口
    """

    def __init__(
        self,
        nodes=None,
        edges=None,
        fencegan_cfg: dict | None = None,
        window: int = 128,
        stride: int = 32,
        gamma: float = 0.5,
        lambda_disp: float = 1.0,
        z_dim: int = 64,
        device: str = "cuda",
    ):
        # ===== 拓扑信息（仅用于前端可视化） =====
        self.nodes = nodes or []
        self.edges = edges or []

        # ===== Fence-GAN 配置解析 =====
        cfg = fencegan_cfg or {}

        # ---- 兼容前端旧字段名 ----
        self.gamma = float(
            cfg.get("gamma")
            or cfg.get("boundary_gamma")
            or gamma
        )
        self.lambda_disp = float(
            cfg.get("lambda_disp")
            or cfg.get("dispersion_lambda")
            or lambda_disp
        )
        self.z_dim = int(
            cfg.get("z_dim")
            or cfg.get("latent_dim")
            or z_dim
        )
        self.lr_G = float(cfg.get("lr_G") or cfg.get("lr") or 1e-4)
        self.lr_D = self.lr_G / 5   # ✅ 判别器学习率略小，稳定训练
        self.epochs = int(cfg.get("epochs") or 10)
        self.use_first_pct = int(cfg.get("use_first_pct") or 40)

        self.window = int(cfg.get("window", window))
        self.stride = int(cfg.get("stride", stride))
        self.device = cfg.get("device", device)

        # ===== 模型与数据容器 =====
        self.model: FenceGAN | None = None
        self.scaler = None  # (mean, std)

        # 日志输出
        print(f"[Engine Init] γ={self.gamma}, λ={self.lambda_disp}, z_dim={self.z_dim}, "
              f"lr_G={self.lr_G}, lr_D={self.lr_D}, epochs={self.epochs}, use_first_pct={self.use_first_pct}")

    # -------------------------
    # 标准化
    # -------------------------
    @staticmethod
    def _fit_scaler(X: np.ndarray):
        mu = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return mu, std

    @staticmethod
    def _apply_scaler(X: np.ndarray, scaler):
        mu, std = scaler
        return (X - mu) / std

    # -------------------------
    # 训练 + 检测主逻辑
    # -------------------------
    def fit_predict(
        self,
        data: pd.DataFrame | np.ndarray,
        labels: np.ndarray | None = None,
        batch_size: int = 128,
    ):
        if isinstance(data, pd.DataFrame):
            X = data.values.astype(np.float32)
        else:
            X = np.asarray(data, dtype=np.float32)
        T, C = X.shape

        # ---- 归一化预处理 ----
        X = np.nan_to_num(X)
        X = np.clip(X, -10, 10)

        # ---- 使用前 use_first_pct% 训练 ----
        train_len = max(10, int(T * self.use_first_pct / 100))
        X_train = X[:train_len]

        # ---- 标准化 ----
        self.scaler = self._fit_scaler(X_train)
        X_std = self._apply_scaler(X, self.scaler)

        # ---- 滑窗化 ----
        Xw = make_windows_from_matrix(X_std, self.window, self.stride)
        Nw = Xw.shape[0]
        if Nw == 0:
            raise ValueError("窗口数量为 0，请检查 window/stride。")
        Xw_flat = Xw.reshape(Nw, -1)

        # ---- 初始化 Fence-GAN ----
        x_dim = Xw_flat.shape[1]
        self.model = FenceGAN(
            x_dim=x_dim,
            z_dim=self.z_dim,
            gamma=self.gamma,
            lambda_disp=self.lambda_disp,
            lr_G=self.lr_G,
            lr_D=self.lr_D,
            device=self.device,
        )

        # ✅ 梯度稳定策略：权重裁剪 & 梯度裁剪
        for p in self.model.D.parameters():
            p.data.clamp_(-0.01, 0.01)

        print(f"[Training Start] X_train={Xw_flat.shape}, epochs={self.epochs}")
        self.model.fit(Xw_flat, epochs=self.epochs, batch_size=batch_size)

        # ✅ 再次梯度裁剪，防止最后几轮爆炸
        torch.nn.utils.clip_grad_norm_(self.model.G.parameters(), max_norm=10)
        torch.nn.utils.clip_grad_norm_(self.model.D.parameters(), max_norm=10)

        print("[Training Done]")

        # ---- 推断异常概率 ----
        w_scores = self.model.score(Xw_flat)
        ts_scores = window_scores_to_time_series(w_scores, T, self.window, self.stride)

        return {
            "anomaly_prob": ts_scores,
            "window_prob": w_scores,
            "window": self.window,
            "stride": self.stride,
            "gamma": self.gamma,
            "nodes": self.nodes,
            "edges": self.edges,
            "epochs": self.epochs,
            "lr": self.lr_G,
        }

    # -------------------------
    # ✅ 前端兼容接口 detect()
    # -------------------------
    def detect(self, data, context=None):
        """
        兼容旧接口。context 参数可传入阈值、元信息等。
        内部直接调用 fit_predict()。
        """
        ctx = context or {}
        result = self.fit_predict(data)
        result["context"] = ctx
        return result

    # -------------------------
    # 策略建议
    # -------------------------
    def recommend_actions(self, result: dict) -> str:
        mp = float(np.mean(result.get("anomaly_prob", [0.0])))
        if mp < 0.30:
            return "系统稳定；维持当前策略，按周轮换模型。"
        elif mp < 0.60:
            return "疑似入侵；建议提高阈值(+0.05)并启用拟态随机化（中）。"
        elif mp < 0.80:
            return "FDI 风险较高；建议缩短检测窗口(-25%)并对高热度通道二次校验。"
        else:
            return "高危；建议启用应急隔离与拟态迁移，限制高热通道。"
