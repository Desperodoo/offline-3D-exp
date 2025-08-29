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


def gumbel_noise(
    shape: Union[Tuple[int, ...], torch.Size], 
    device: Optional[torch.device] = None, 
    eps: float = 1e-10
) -> torch.Tensor:
    """生成服从Gumbel(0, 1)分布的随机噪声。
    
    Gumbel分布常用于实现Gumbel-Softmax重参数化技巧，可以对离散分布进行可微分采样。
    该噪声用于将确定性的argmax变换为随机采样操作。
    
    参数
    ----
    shape : tuple 或 torch.Size
        输出张量的形状
    device : torch.device, 可选
        张量所在的设备
    eps : float, 默认=1e-10
        用于数值稳定性的小常数，防止对0取对数
        
    返回值
    ----
    torch.Tensor
        形状为shape的Gumbel噪声张量
    """
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def sample_action_indices(
    action_logits: torch.Tensor, 
    num_valid_actions: torch.Tensor, 
    stochastic: bool = True,
    temperature: float = 1.0,
    off_value: float = -1e6  # 修改默认值为更严格的mask
) -> torch.Tensor:
    """基于logits采样或选择动作索引，同时尊重有效动作掩码。
    
    该函数支持两种模式：
    1. 随机模式（默认）：使用Gumbel-Max技巧进行采样，等效于按概率采样但可微分
    2. 确定性模式：直接选择最高logit值的动作索引
    
    在两种模式下，都会强制保证只从有效动作中选择，即索引必须小于对应批次的num_valid_actions值。
    
    参数
    ----
    action_logits : torch.Tensor [B, N]
        每个批次的动作logits，B为批次大小，N为可能的动作总数
    num_valid_actions : torch.Tensor [B]
        每个批次中有效动作的数量，用于掩盖无效动作
    stochastic : bool, 默认=True
        是否使用随机采样（True）或确定性选择（False）
    temperature : float, 默认=1.0
        采样温度，控制随机性程度。较低的温度会使分布更加尖锐，较高的温度使分布更加平缓
        
    返回值
    ----
    torch.Tensor [B]
        每个批次选择的动作索引
    """
    # 参数验证
    # 增加对温度和有效动作数的验证
    if temperature <= 0:
        raise ValueError(f"温度应为正数，但接收到{temperature}")
        
    if torch.any(num_valid_actions <= 0):
        raise ValueError("有效动作数必须为正整数")
    
    if torch.any(num_valid_actions > action_logits.size(1)):
        raise ValueError(f"有效动作数不能超过动作总数{action_logits.size(1)}")

    if action_logits.dim() != 2:
        raise ValueError(f"action_logits应为2D张量 [批次大小, 动作数], 但形状为: {action_logits.shape}")
    
    if num_valid_actions.dim() != 1 or action_logits.size(0) != num_valid_actions.size(0):
        raise ValueError(f"num_valid_actions应为1D张量且长度与批次大小相同，当前形状: {num_valid_actions.shape}")
    
    # 应用温度调节
    if temperature != 1.0:
        scaled_logits = action_logits / max(temperature, 1e-8)  # 防止除以零
    else:
        scaled_logits = action_logits
    
    # 随机模式：添加Gumbel噪声实现随机采样
    if stochastic:
        noise = gumbel_noise(scaled_logits.shape, device=scaled_logits.device)
        perturbed_logits = scaled_logits + noise
    else:
        perturbed_logits = scaled_logits
        perturbed_logits += 1e-6 * torch.randn_like(perturbed_logits)  # 添加微小噪声以避免平局情况
    
    # 创建有效动作掩码，使无效动作概率为负无穷
    batch_size, max_actions = action_logits.size()
    indices_range = torch.arange(max_actions, device=action_logits.device).unsqueeze(0).expand(batch_size, -1)
    mask = indices_range >= num_valid_actions.unsqueeze(1)
    
    # 应用掩码并获取最大值索引
    masked_logits = torch.where(
        mask, 
        torch.full_like(perturbed_logits, off_value), 
        perturbed_logits
    )
    
    return torch.argmax(masked_logits, dim=1)


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

def gumbel_softmax_sample(
    action_logits: torch.Tensor,
    num_valid_actions: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
    off_value: float = -1e6  # 修改默认值为更严格的mask
) -> torch.Tensor:
    """使用Gumbel-Softmax进行可微分的动作采样。
    
    该函数实现Gumbel-Softmax重参数化技巧，允许梯度通过离散采样操作反向传播。
    适用于训练时需要可微分采样的场景，如actor-critic方法中的策略梯度。
    
    参数
    ----
    action_logits : torch.Tensor [B, N]
        每个批次的动作logits，B为批次大小，N为可能的动作总数
    num_valid_actions : torch.Tensor [B]
        每个批次中有效动作的数量，用于掩盖无效动作
    temperature : float, 默认=1.0
        Gumbel-Softmax温度参数，控制采样的"硬度"
        较低的温度使分布更接近one-hot，较高的温度使分布更平滑
    hard : bool, 默认=False
        是否使用直通估计器(straight-through estimator)
        True时前向传播返回one-hot，反向传播使用soft概率
    off_value : float, 默认=-1e6  # 修改默认值为更严格的mask
        无效动作位置的填充值
        
    返回值
    ----
    torch.Tensor
        如果hard=False，返回软概率分布 [B, N]
        如果hard=True，返回one-hot向量 [B, N]，但梯度通过软分布计算
    """
    # 参数验证
    if temperature <= 0:
        raise ValueError(f"温度应为正数，但接收到{temperature}")
        
    if torch.any(num_valid_actions <= 0):
        raise ValueError("有效动作数必须为正整数")
    
    if torch.any(num_valid_actions > action_logits.size(1)):
        raise ValueError(f"有效动作数不能超过动作总数{action_logits.size(1)}")

    if action_logits.dim() != 2:
        raise ValueError(f"action_logits应为2D张量 [批次大小, 动作数], 但形状为: {action_logits.shape}")
    
    if num_valid_actions.dim() != 1 or action_logits.size(0) != num_valid_actions.size(0):
        raise ValueError(f"num_valid_actions应为1D张量且长度与批次大小相同，当前形状: {num_valid_actions.shape}")
    
    # 创建有效动作掩码
    batch_size, max_actions = action_logits.size()
    indices_range = torch.arange(max_actions, device=action_logits.device).unsqueeze(0).expand(batch_size, -1)
    mask = indices_range >= num_valid_actions.unsqueeze(1)
    
    # 应用掩码到logits
    masked_logits = torch.where(
        mask, 
        torch.full_like(action_logits, off_value), 
        action_logits
    )
    
    # 应用温度调节
    scaled_logits = masked_logits / max(temperature, 1e-8)
    
    # 添加Gumbel噪声
    gumbel_noise_tensor = gumbel_noise(scaled_logits.shape, device=scaled_logits.device)
    perturbed_logits = scaled_logits + gumbel_noise_tensor
    
    # 计算softmax概率
    soft_sample = F.softmax(perturbed_logits, dim=-1)
    
    if hard:
        # 直通估计器：前向传播使用one-hot，反向传播使用soft
        # 获取最大值索引
        _, max_indices = torch.max(perturbed_logits, dim=-1, keepdim=True)
        # 创建one-hot
        hard_sample = torch.zeros_like(soft_sample).scatter_(-1, max_indices, 1.0)
        # 直通技巧：hard - soft.detach() + soft
        return hard_sample - soft_sample.detach() + soft_sample
    else:
        return soft_sample

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
