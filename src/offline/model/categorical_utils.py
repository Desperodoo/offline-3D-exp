"""
categorical_utils.py
-------------------------------------------------------------
Utility functions for **discrete action** logits / indices.
Originally written for diffusion‑LM as *token* utilities, this
refactor renames the public API to action‑centric terms so that
it reads naturally in RL settings.

Backward‑compatibility:
* Legacy aliases with the old *token* names are kept for now and
  emit DeprecationWarning.  They can be removed once the entire
  codebase has migrated.
"""

from __future__ import annotations

import warnings
from typing import Union, Tuple, Optional, Sequence

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 基础工具函数
# -----------------------------------------------------------------------------

def log_prob_from_logits(action_logits: torch.Tensor) -> torch.Tensor:
    """将logits转换为对数概率（沿最后一维）。
    
    通过减去logsumexp实现数值稳定的softmax对数概率计算。
    该函数在计算概率分布的对数似然时非常有用。
    
    参数
    ----
    action_logits : torch.Tensor
        任意形状的logits张量，最后一维表示类别
        
    返回值
    ----
    torch.Tensor
        与输入相同形状的对数概率张量，最后一维上和为0（概率和为1）
    """
    return action_logits - action_logits.logsumexp(dim=-1, keepdim=True)


def one_hot(
    action_indices: torch.Tensor, 
    num_classes: int, 
    validate_indices: bool = True
) -> torch.Tensor:
    """将离散动作索引转换为one-hot浮点张量。
    
    将整数索引转换为独热编码，使每个索引表示为仅一个维度为1，其余为0的向量。
    输出张量的最后一维大小为num_classes。
    
    参数
    ----
    action_indices : torch.Tensor
        整数动作索引张量，任意形状
    num_classes : int
        one-hot编码的类别数量（向量维度）
    validate_indices : bool, 默认=True
        是否验证动作索引是否在有效范围内
        
    返回值
    ----
    torch.Tensor
        输入张量形状 + [num_classes] 的浮点型one-hot张量
    """
    # 验证索引在有效范围内
    if validate_indices:
        max_idx = action_indices.max().item()
        if max_idx >= num_classes:
            raise ValueError(f"动作索引超出有效范围: 最大索引 {max_idx} >= 类别数量 {num_classes}")
    
    return F.one_hot(action_indices, num_classes=num_classes).float()

# -----------------------------------------------------------------------------
# 与扩散模型交互的接口
# -----------------------------------------------------------------------------

def actions_to_logits(
    action_indices: torch.Tensor,
    num_classes: int,
    on_value: float = 0.0,
    off_value: float = -1e6,  # 修改默认值为更严格的mask
) -> torch.Tensor:
    """将离散动作索引映射为扩散模型使用的logits掩码。
    
    此函数创建一个特殊的logits表示，使每个有效动作在其位置上为on_value，
    在其他位置为off_value。这种极端的logits分布在反向扩散过程中用于
    条件生成，控制特定位置的动作分布。
    
    off_value默认设为-64.0，实质上将这些位置的概率设为0。
    
    参数
    ----
    action_indices : torch.Tensor
        离散动作索引
    num_classes : int
        动作空间大小（类别总数）
    on_value : float, 默认=0.0
        有效动作位置的logit值
    off_value : float, 默认=-1e6
        非有效动作位置的logit值，设为极小值
        
    返回值
    ----
    torch.Tensor
        logit掩码，形状为输入张量形状 + [num_classes]
    """
    # 将任何-1或负数索引（通常是填充标记）替换为0
    clamped_indices = action_indices.clamp(min=0)
    
    # 转换为one-hot张量
    oh = one_hot(clamped_indices, num_classes, validate_indices=False)
    
    # 创建目标logits
    logits = torch.where(
        oh == 1, 
        torch.full_like(oh, on_value), 
        torch.full_like(oh, off_value)
    )
    
    return logits
