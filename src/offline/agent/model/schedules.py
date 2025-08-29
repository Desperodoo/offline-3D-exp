"""
schedules.py
-------------------------------------------------------------
Diffusion **β‑schedule** & noise injection utilities for discrete
*action indices* (offline RL, graph‑state setting).

Originally written for language‑token diffusion, this refactor
switches all public names to the *action* vocabulary while keeping
thin, deprecated aliases so existing calls continue to work until
the rest of the codebase is migrated.
"""

from __future__ import annotations

import math
import warnings
import torch
import numpy as np
from .categorical_utils import actions_to_logits
# -----------------------------------------------------------------------------
# β‑schedules (forward diffusion)
# -----------------------------------------------------------------------------
# schedules.py
import torch



def cosine_action_schedule(
    T: int,
    s: float = 0.008,
    *,
    rescale_to: float | None = None,
) -> torch.Tensor:
    """
    生成长度为 T 的余弦 β 调度序列 (β_1 … β_T)。
    
    余弦调度避免时间步采样集中在前几步，提供更平滑的噪声调度。
    
    默认行为
    --------
    * 与 linear_action_schedule 保持一致的缩放行为
    * 若确需兼容旧代码，可设置 `rescale_to=1.0`
    
    参数
    ----
    T : int
        扩散时间步个数 (T ≥ 1)
    s : float, default=0.008
        小的偏移量，防止 beta 在 t=0 时为 0
    rescale_to : float | None, default=None
        若为正数，则按 `rescale_to / Σβ` 重新缩放整条序列；
        设为 1.0 可复刻旧版 "Σβ≈1" 行为。
    
    返回
    ----
    betas : torch.Tensor  shape [T]
    """
    if T < 1:
        raise ValueError("T must be positive")
    if s <= 0:
        raise ValueError("s must be positive")
    
    steps = T + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 0, 0.999)
    
    # 转换为 torch.Tensor
    betas = torch.from_numpy(betas).float()
    
    if rescale_to is not None:
        if rescale_to <= 0:
            raise ValueError("rescale_to must be positive")
        betas = betas * (rescale_to / betas.sum())
    
    return betas

# -----------------------------------------------------------------------------

def linear_action_schedule(
    T: int,
    beta_start: float = 0.05,
    beta_end: float   = 0.9,
    *,
    rescale_to: float | None = None,
) -> torch.Tensor:
    """
    生成长度为 T 的线性 β 调度序列 (β_1 … β_T)。

    默认行为
    --------
    * **不**再把 Σβ 缩放到 1；这样 T 越大，总噪声自然增大。
      - 对离散 logits 扩散更符合 “多步粗→细” 直觉；
      - 避免 T=20 时每步 β≈0.01 噪声过弱、一两步即可去噪完。
    * 若确需兼容旧代码，可设置 `rescale_to=1.0`。

    参数
    ----
    T : int
        扩散时间步个数 (T ≥ 1)
    beta_start, beta_end : float
        线性序列端点 (0 < beta_start < beta_end < 1)
    rescale_to : float | None, default=None
        若为正数，则按 `rescale_to / Σβ` 重新缩放整条序列；
        设为 1.0 可复刻旧版 “Σβ≈1” 行为。

    返回
    ----
    betas : torch.Tensor  shape [T]
    """
    if not (0.0 < beta_start < beta_end < 1.0):
        raise ValueError(
            f"Require 0 < beta_start < beta_end < 1, got {beta_start=}, {beta_end=}."
        )
    if T < 1:
        raise ValueError("T must be positive")

    betas = torch.linspace(beta_start, beta_end, steps=T)

    if rescale_to is not None:
        if rescale_to <= 0:
            raise ValueError("rescale_to must be positive")
        betas = betas * (rescale_to / betas.sum())

    return betas


# -----------------------------------------------------------------------------
# Forward process q(xₜ | x₀) for discrete actions
# -----------------------------------------------------------------------------
def q_sample_logits(action_indices_start: torch.Tensor,
                    t: torch.Tensor,
                    betas: torch.Tensor,
                    padding_mask: torch.Tensor,
                    off_value: float = -64.,
                    init_scale: float = 4.0) -> torch.Tensor:
    """
    输入:
        action_indices0  [B]     — 干净动作索引
        t                [B]     — 扩散步
        betas            [T]     — β 调度
        padding_mask   [B,1,N]   — 1=填充/无效
    输出:
        logits_t       [B,N]     — t 步带噪 logits
    """
    # 1. one-hot → logits_0
    N = padding_mask.size(2)
    logits0 = actions_to_logits(action_indices_start, N,
                                on_value=0., off_value=off_value)  # [B,N]

    # 2. 注入连续噪声
    beta = betas[t].view(-1, 1)                      # [B,1]
    noise = torch.randn_like(logits0) * init_scale   # σ≈4 ⇒ tanh≈±0.7
    # 2.5 做 “线性插值” :   logits_{t-1} = (1-βₜ) · logits_t  + βₜ · pred_logits
    #     或 “sqrt插值”：   logits_{t-1} = (1-βₜ).sqrt() · logits_t  + βₜ.sqrt() · pred_logits
    #     二选一，这里采用 “线性插值”
    logits_t = (1.0 - beta) * logits0 + beta * noise

    # 3. mask 无效动作
    logits_t = logits_t.masked_fill(padding_mask.squeeze(1).bool(), off_value)
    return logits_t
