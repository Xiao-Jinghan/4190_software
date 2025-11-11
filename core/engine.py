# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fence_gan import FenceGAN
from .utils import (
    make_windows_from_matrix,
    window_scores_to_time_series,
)

class _AttackClassifier(nn.Module):
    """
    轻量级窗口级多分类器（全连接 + ReLU + Softmax），用于预测 KDD99 攻击类型。
    输入：展平窗口向量
    输出：各类别的 logits
    """
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, n_classes)
        )

    def forward(self, x):
        return self.net(x)

def _window_class_to_time_series(class_probs_w: np.ndarray, T: int, window: int, stride: int) -> np.ndarray:
    """
    将窗口级类别概率 (Nw, K) 映射到逐时刻 (T, K)，用窗口覆盖平均。
    """
    class_probs_w = np.asarray(class_probs_w, dtype=np.float32)
    Nw, K = class_probs_w.shape
    ts = np.zeros((T, K), dtype=np.float32)
    ct = np.zeros(T, dtype=np.float32)

    idx = 0
    for start in range(0, T - window + 1, stride):
        ts[start:start + window, :] += class_probs_w[idx][None, :]
        ct[start:start + window] += 1.0
        idx += 1

    ct_safe = np.where(ct == 0.0, 1.0, ct)[:, None]
    ts = ts / ct_safe
    return ts

class MimicDefenseEngine:
    """
    拟态防御核心引擎：
      - 支持前端动态传入 nodes / edges / fencegan_cfg
      - fencegan_cfg 可兼容旧版字段（latent_dim、boundary_gamma、dispersion_lambda 等）
      - detect() 为统一外部接口
      - ✅ 新增：攻击类型多分类（基于 KDD99 的 attack_code）
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

        # 新增：多分类器（窗口级）
        self.cls_head: _AttackClassifier | None = None
        self._cls_classes = None  # np.ndarray of class codes, length K

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
        *,
        attack_codes: np.ndarray | None = None,
        code_to_name: dict | None = None,
        batch_size: int = 128,
    ):
        """
        labels: 可选，逐时刻二值标签(0/1)；若提供将用于更健康的训练采样
        attack_codes: 可选，逐时刻多分类编码；若提供则训练窗口级多分类器
        code_to_name: 可选，类别显示名映射（前端绘图/报告）
        """
        if isinstance(data, pd.DataFrame):
            X = data.values.astype(np.float32)
        else:
            X = np.asarray(data, dtype=np.float32)
        T, C = X.shape

        # 兼容：如果最后两列是 label/attack_code，我们并不在此处删除它们，
        # 以避免破坏既有流程；但后续窗口构造与训练会按 use_first_pct 选取正常窗口。
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
        x_dim = Xw_flat.shape[1]

        # ---- Fence-GAN：仅用“正常窗口”训练（若给出 labels 则更严格筛选）
        normal_mask_w = np.ones(Nw, dtype=bool)
        if labels is not None:
            # 窗口视角：只保留窗口内全 0 的作为“正常窗口”
            lab_w = []
            for s in range(0, T - self.window + 1, self.stride):
                seg = labels[s: s + self.window]
                lab_w.append(int(np.all(seg == 0)))
            lab_w = np.asarray(lab_w, dtype=bool)
            if lab_w.shape[0] == Nw:
                normal_mask_w = lab_w

        Xw_train = Xw_flat[normal_mask_w]

        self.model = FenceGAN(
            x_dim=x_dim,
            z_dim=self.z_dim,
            gamma=self.gamma,
            lambda_disp=self.lambda_disp,
            lr_G=self.lr_G,
            lr_D=self.lr_D,
            device=self.device,
        )

        # ✅ 权重裁剪（稳定）
        for p in self.model.D.parameters():
            p.data.clamp_(-0.01, 0.01)

        print(f"[Training Start] X_train={Xw_train.shape}, epochs={self.epochs}")
        self.model.fit(Xw_train, epochs=self.epochs, batch_size=batch_size)

        # ✅ 再次梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.G.parameters(), max_norm=10)
        torch.nn.utils.clip_grad_norm_(self.model.D.parameters(), max_norm=10)
        print("[Training Done]")

        # ---- Fence-GAN 推断异常概率 ----
        w_scores = self.model.score(Xw_flat)
        ts_scores = window_scores_to_time_series(w_scores, T, self.window, self.stride)

        # ---- （新增）攻击类型多分类：若提供 attack_codes，则进行窗口级训练并投射到时序 ----
        cls_result = {
            "classes": [],
            "class_names": [],
            "type_prob_ts": None,       # (T, K)
            "type_pred_codes_ts": None  # (T,)
        }
        if attack_codes is not None:
            attack_codes = np.asarray(attack_codes).reshape(-1)
            # 构造窗口级标签：每个窗口用“众数”编码
            lab_w = []
            for s in range(0, T - self.window + 1, self.stride):
                seg = attack_codes[s: s + self.window]
                # 众数（兼顾异常窗）
                vals, cnts = np.unique(seg, return_counts=True)
                lab_w.append(int(vals[np.argmax(cnts)]))
            lab_w = np.asarray(lab_w, dtype=int)

            # 类别集合与映射到 [0..K-1]
            classes = np.unique(attack_codes)
            classes_sorted = np.sort(classes)
            code2idx = {c: i for i, c in enumerate(classes_sorted)}
            idx2code = {i: c for c, i in code2idx.items()}
            y_w = np.array([code2idx[c] for c in lab_w], dtype=int)
            K = len(classes_sorted)

            # 训练简单分类头
            self.cls_head = _AttackClassifier(in_dim=x_dim, n_classes=K).to(self.device)
            opt = torch.optim.Adam(self.cls_head.parameters(), lr=1e-3)
            ce = nn.CrossEntropyLoss()

            Xw_t = torch.from_numpy(Xw_flat).float().to(self.device)
            y_w_t = torch.from_numpy(y_w).long().to(self.device)

            epochs_cls = max(5, min(30, self.epochs // 2))
            self.cls_head.train()
            for ep in range(1, epochs_cls + 1):
                # 这里使用整批训练即可（数据量较大也可改 DataLoader）
                logits = self.cls_head(Xw_t)
                loss = ce(logits, y_w_t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if ep % 5 == 0:
                    print(f"[Cls Epoch {ep}/{epochs_cls}] loss={float(loss.item()):.4f}")

            # 推断窗口级类别概率
            self.cls_head.eval()
            with torch.no_grad():
                logits = self.cls_head(Xw_t)
                probs_w = F.softmax(logits, dim=1).detach().cpu().numpy()  # (Nw, K)

            # 投射至时间序列 (T, K)
            prob_ts = _window_class_to_time_series(probs_w, T, self.window, self.stride)
            # 逐时刻取 argmax（类别 idx）
            pred_idx_ts = np.argmax(prob_ts, axis=1)  # (T,)
            pred_code_ts = np.array([idx2code[i] for i in pred_idx_ts], dtype=int)

            # 类别名
            class_names = []
            if code_to_name:
                class_names = [str(code_to_name.get(int(c), str(int(c)))) for c in classes_sorted]
            else:
                class_names = [str(int(c)) for c in classes_sorted]

            cls_result = {
                "classes": classes_sorted.tolist(),
                "class_names": class_names,
                "type_prob_ts": prob_ts,
                "type_pred_codes_ts": pred_code_ts
            }

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
            # 新增输出：
            "type_classes": cls_result["classes"],
            "type_class_names": cls_result["class_names"],
            "type_prob_ts": cls_result["type_prob_ts"],               # (T, K) or None
            "type_pred_codes_ts": cls_result["type_pred_codes_ts"],   # (T,) or None
        }

    # -------------------------
    # ✅ 前端兼容接口 detect()
    # -------------------------
    def detect(self, data, context=None):
        """
        兼容旧接口。context 参数可传入阈值、元信息等。
        现在支持：
          context["binary_labels"] : 可选，逐时刻 0/1
          context["attack_codes"]  : 可选，逐时刻 attack_code
          context["attack_code_to_name"] : 可选，类别名映射
        """
        ctx = context or {}
        result = self.fit_predict(
            data,
            labels=np.asarray(ctx.get("binary_labels")) if ctx.get("binary_labels") is not None else None,
            attack_codes=np.asarray(ctx.get("attack_codes")) if ctx.get("attack_codes") is not None else None,
            code_to_name=ctx.get("attack_code_to_name"),
        )
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
