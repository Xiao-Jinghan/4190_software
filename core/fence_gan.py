# core/fence_gan.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================
# 公用
# ======================
def to_device(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))
    return x.to(device)


def logit(p: float) -> float:
    """Sigmoid 的逆函数，避免 0/1 上的数值问题。"""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(math.log(p / (1.0 - p)))


# ======================
# 🔩 模型定义
# ======================
class Generator(nn.Module):
    """简单 MLP 生成器：z -> x_fake"""
    def __init__(self, z_dim: int, x_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden // 2, x_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """判别器输出**logits**（无 Sigmoid！）"""
    def __init__(self, x_dim: int, hidden: int = 256, dropout: float = 0.1, spectral_norm: bool = False):
        super().__init__()
        def maybe_sn(layer: nn.Module) -> nn.Module:
            return nn.utils.spectral_norm(layer) if spectral_norm else layer

        self.net = nn.Sequential(
            maybe_sn(nn.Linear(x_dim, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            maybe_sn(nn.Linear(hidden, hidden // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden // 2, 1),   # ← 无 Sigmoid
        )

    def forward(self, x):
        return self.net(x).view(-1)  # logits 形状: (B,)


# ======================
# 🎯 Fence-GAN
# ======================
class FenceGAN:
    """
    训练目标（生成器）：
      1) 让 D(G(z)) 的 **logits** 靠近 logit(gamma) —— 在边界上“贴边”；
      2) 轻量的 dispersion 正则，防止模式坍塌；
    判别器：BCEWithLogitsLoss，正样本=真实(正常)，负样本=生成。

    推断：返回 **异常概率 = 1 - sigmoid(D(x))**。
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int = 64,
        gamma: float = 0.5,
        lambda_disp: float = 1.0,
        g_hidden: int = 256,
        d_hidden: int = 256,
        spectral_norm: bool = True,
        lr_G: float = 2e-4,
        lr_D: float = 1e-4,
        betas=(0.5, 0.9),
        n_critic: int = 3,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        dropout_G: float = 0.0,
        dropout_D: float = 0.1,
        seed: int | None = 42,
    ):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.gamma = float(gamma)
        self.gamma_logit = logit(self.gamma)
        self.lambda_disp = float(lambda_disp)
        self.n_critic = int(n_critic)
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.G = Generator(z_dim, x_dim, hidden=g_hidden, dropout=dropout_G).to(self.device)
        self.D = Discriminator(x_dim, hidden=d_hidden, dropout=dropout_D, spectral_norm=spectral_norm).to(self.device)

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr_G, betas=betas)
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr_D, betas=betas)
        self.bce = nn.BCEWithLogitsLoss()

    # ---------- utils ----------
    def _sample_z(self, n: int) -> torch.Tensor:
        # 高斯噪声
        return torch.randn(n, self.z_dim, device=self.device)

    @staticmethod
    def _dispersion(x_fake: torch.Tensor) -> torch.Tensor:
        """
        计算简单 dispersion：距离样本中心的均值距离的倒数，越分散 loss 越小。
        """
        center = x_fake.mean(dim=0, keepdim=True)
        dist = torch.sqrt(torch.sum((x_fake - center) ** 2, dim=1) + 1e-8)
        avg_distance = torch.mean(dist)
        return 1.0 / (avg_distance + 1e-8)

    # ---------- API ----------
    def fit(self, X_normal: np.ndarray, epochs: int = 50, batch_size: int = 128, log_every: int = 1):
        """
        仅用“正常样本”训练（半监督/一类）。
        X_normal: (N, x_dim)
        """
        X_normal = np.asarray(X_normal, dtype=np.float32)
        assert X_normal.ndim == 2 and X_normal.shape[1] == self.x_dim

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_normal))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        target_logit = torch.tensor(self.gamma_logit, device=self.device)

        for epoch in range(1, epochs + 1):
            running_d, running_g, steps = 0.0, 0.0, 0

            for xb, in loader:
                xb = xb.to(self.device)

                # ----- 训练 D：多步 -----
                for _ in range(self.n_critic):
                    z = self._sample_z(xb.size(0))
                    x_fake = self.G(z).detach()                # 阻断到 G

                    d_real = self.D(xb)                        # logits
                    d_fake = self.D(x_fake)                    # logits
                    y_real = torch.ones_like(d_real)
                    y_fake = torch.zeros_like(d_fake)

                    d_loss = self.bce(d_real, y_real) + self.bce(d_fake, y_fake)

                    self.opt_D.zero_grad(set_to_none=True)
                    d_loss.backward()
                    self.opt_D.step()

                # ----- 训练 G：一步 -----
                z = self._sample_z(xb.size(0))
                x_fake = self.G(z)
                d_fake_logits = self.D(x_fake)

                # (1) fence：让 logits 靠近 logit(gamma)
                g_fence = torch.mean(torch.abs(d_fake_logits - target_logit))

                # (2) dispersion：防止坍塌
                g_disp = self._dispersion(x_fake) * self.lambda_disp

                g_loss = g_fence + g_disp

                self.opt_G.zero_grad(set_to_none=True)
                g_loss.backward()
                self.opt_G.step()

                running_d += float(d_loss.item())
                running_g += float(g_loss.item())
                steps += 1

            if epoch % log_every == 0:
                print(
                    f"[Epoch {epoch:03d}/{epochs}] "
                    f"D_loss={running_d/steps:.4f}  G_loss={running_g/steps:.4f}  "
                    f"(fence={g_fence.item():.4f}, disp={g_disp.item():.4f})"
                )

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        返回“异常概率” (越大越异常) = 1 - sigmoid(D(x))。
        """
        self.D.eval()
        X = to_device(X, self.device).float()
        logits = self.D(X)                      # P(正常) 的 logits
        p_normal = torch.sigmoid(logits)
        p_anom = 1.0 - p_normal
        return p_anom.detach().cpu().numpy().reshape(-1)

    @torch.no_grad()
    def normal_prob(self, X: np.ndarray) -> np.ndarray:
        """P(正常)"""
        self.D.eval()
        X = to_device(X, self.device).float()
        logits = self.D(X)
        return torch.sigmoid(logits).cpu().numpy().reshape(-1)
