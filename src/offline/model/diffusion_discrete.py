from __future__ import annotations
from typing import Optional
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.schedules import (
    linear_action_schedule,
    cosine_action_schedule,
    q_sample_logits,
)


class DiscreteDiffusion(nn.Module):
    """离散扩散模型，用于离散动作空间的生成和学习。
    
    该模型实现了离散扩散概率模型(DDPM)，特别适用于离散动作空间。
    模型通过一系列的前向扩散步骤向干净动作添加噪声，然后学习
    逆向扩散过程以从噪声中恢复干净动作。

    参数
    ----------
    num_actions : int
        离散动作空间大小(N)。对于简化模式，这是最大动作数。
    model : nn.Module
        去噪网络，将输入映射 ``(logits_t, t, obs)`` → ``action_logits``。
    T : int, 默认值 20
        扩散步骤数量。
    schedule : {"linear", "cosine"}
        Beta调度类型，控制噪声添加速率。
    """

    def __init__(
        self,
        num_actions: int,
        model: nn.Module,
        T: int = 20,
        schedule: str = "linear",
    ) -> None:
        super().__init__()
        
        # 参数验证
        if num_actions <= 0:
            raise ValueError(f"num_actions必须为正整数，得到{num_actions}")
        if T < 0:  # 修改：允许T=0，退化成普通DQL
            raise ValueError(f"扩散步骤T必须为非负整数，得到{T}")
            
        self.num_actions = num_actions
        self.model = model
        self.T = T             # 超参数，可调
        self.init_scale = 3.0  # 超参数，可调
        
        # 根据schedule类型设置beta序列
        if T == 0:
            # T=0时，退化成普通DQL，不需要beta序列
            betas = torch.empty(0)  # 空张量
        elif schedule == "linear":
            betas = linear_action_schedule(T)
        elif schedule == "cosine":
            betas = cosine_action_schedule(T)
        else:
            raise NotImplementedError(f"{schedule}调度类型尚未实现")

        self.register_buffer("betas", betas)  # 形状 [T] 或 [0]

    # ------------------------------------------------------------------
    # 训练相关方法
    # ------------------------------------------------------------------
    def loss(self, action_indices0: torch.Tensor, padding_mask: torch.Tensor, obs) -> tuple:
        """计算去噪网络预测与真实动作之间的损失，支持两步损失。
        
        该函数执行以下步骤：
        1. 随机选择扩散时间步t（如果T=0则跳过扩散过程）
        2. 对干净动作进行前向扩散采样，获得带噪声的动作（T=0时跳过）
        3. 将带噪声动作转换为logits表示
        4. 使用去噪网络预测干净动作
        5. 计算预测与真实动作之间的损失（支持CE + 一致性损失）

        参数
        ----------
        action_indices0 : torch.Tensor
            真实（干净）动作索引，形状 [B]，其中B为批次大小
        padding_mask : torch.Tensor
            标识有效动作(0)和填充位置(1)的掩码，形状 [B, 1, N]
        obs : 任意
            环境特定的条件信息（见去噪网络定义）
            
        返回值
        ----------
        tuple
            (loss, loss_dict): 总损失标量值和损失字典
        """
        # 验证模型输出维度是否与num_actions匹配
        if hasattr(self.model, 'max_actions') and self.model.max_actions != self.num_actions:
            raise ValueError(
                f"模型的max_actions({self.model.max_actions})与扩散模型的num_actions({self.num_actions})不匹配"
            )
        
        # 获取批次大小
        B = action_indices0.size(0)
        
        # T=0时退化成普通DQL，直接使用模型作为策略网络
        if self.T == 0:
            # 直接使用模型预测
            pred_logits = self.model(None, None, obs)
            
            # 计算损失
            loss = F.cross_entropy(pred_logits, action_indices0, reduction="mean")
            
            # 损失字典（T=0时的特殊标记）
            loss_dict = {
                'diff/loss_ce': loss.item(),  # CE损失
            }
            
            return loss, loss_dict
        
        # T>0时的正常扩散过程
        # 随机采样时间步t
        t = torch.randint(0, self.T, (B,), device=action_indices0.device)

        # 1. 前向加噪 (logits-space)
        logits_t = q_sample_logits(action_indices0, t, self.betas, padding_mask, init_scale=self.init_scale)

        # 2. 去噪预测
        pred_logits = self.model(logits_t, t, obs)

        # 3. 计算两步损失：CE损失 + 一致性损失
        # 主要损失：交叉熵 → 等价 KL(one-hot‖softmax(pred))
        loss_ce = F.cross_entropy(pred_logits, action_indices0, reduction="mean")
        
        # 一致性损失：teacher forcing
        loss_consistency = 0.0
        consistency_weight = 0.1  # 一致性损失权重
        
        # if self.T > 1:  # 只有多步扩散时才计算一致性损失
        #     with torch.no_grad():
        #         # 计算t-1时刻的teacher logits
        #         t_prev = torch.clamp(t - 1, min=0)  # 防止t=0时越界

        #         # 原始模式：重新计算t-1时刻的logits
        #         logits_tm1_teacher = q_sample_logits(action_indices0, t_prev, self.betas, padding_mask, init_scale=self.init_scale)
                
        #         # teacher预测
        #         logits_tm1_pred = self.model(logits_tm1_teacher, t_prev, obs)
            
        #     # KL散度一致性损失
        #     log_pred = F.log_softmax(pred_logits, dim=-1)
        #     soft_teacher = F.softmax(logits_tm1_pred, dim=-1)
        #     loss_consistency = F.kl_div(log_pred, soft_teacher, reduction="batchmean")
        
        # 总损失
        total_loss = loss_ce + consistency_weight * loss_consistency
        
        # 4. 计算损失字典
        loss_dict = {
            'diff/t_sample': t.float().mean(),
            'diff/entropy_xt': (-F.softmax(logits_t, dim=-1) * F.log_softmax(logits_t, dim=-1)).sum(dim=-1).mean(),
            'diff/loss_ce': loss_ce.item(),
            'diff/loss_consistency': loss_consistency if isinstance(loss_consistency, float) else loss_consistency.item(),
            'diff/consistency_weight': consistency_weight,
        }
        
        return total_loss, loss_dict

    @torch.no_grad()
    def _p_sample(
        self,
        logits_t: torch.Tensor,
        t: int,
        obs,
        num_valid_actions: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        encode_obs_cache: Optional[tuple] = None,
        off_value: float = -1e6,  # 修改默认值为更严格的mask
    ) -> torch.Tensor:
        """执行单步逆向扩散，从t时刻到t-1时刻。
        
        该函数是逆向扩散过程的核心步骤，实现了从带噪声动作向干净动作的单步转换。
        当T=0时，直接返回模型预测结果。
        
        注意：此函数通过调用_p_sample_differentiable并使用@torch.no_grad()装饰来避免代码重复。
        
        参数
        ----------
        logits_t : torch.Tensor [B, K]
            t时刻的动作logits
        t : int
            当前时间步
        obs : list
            环境观测信息（见去噪网络定义）
        num_valid_actions : torch.Tensor [B], 可选
            每个批次样本的有效动作数量（原始模式）
        action_mask : torch.Tensor [B, num_actions], 可选
            动作有效性掩码（简化模式）
        encode_obs_cache : Optional[tuple], 默认None
            预计算的图嵌入，用于避免重复计算
        off_value : float, 默认-64.0
            用于填充无效动作位置的值
            
        返回值
        ----------
        torch.Tensor
            t-1时刻的预测logits（T=0时直接返回预测结果）
        """
        # 直接调用可微分版本，@torch.no_grad()装饰器确保梯度不会传播
        return self._p_sample_differentiable(
            logits_t=logits_t,
            t=t,
            obs=obs,
            num_valid_actions=num_valid_actions,
            action_mask=action_mask,
            encode_obs_cache=encode_obs_cache,
            off_value=off_value
        )

    def _sample_core(
        self,
        obs,
        padding_mask: torch.Tensor,
        return_intermediate: bool = False,
        off_value: float = -1e6,
    ):
        """执行扩散采样的核心逻辑，返回最终的logits和中间轨迹。
        
        这是 sample 和 sample_differentiable 的共享核心实现。
        
        参数
        ----------
        obs : list
            环境观测信息元组（见去噪网络定义）
        padding_mask : torch.Tensor [B, 1, N]
            标识有效动作(0)和填充位置(1)的掩码
        return_intermediate : bool, 默认False
            是否返回中间采样结果
        off_value : float, 默认-64.0
            用于填充无效动作位置的值
            
        返回值
        ----------
        tuple
            (final_logits, traj): 最终logits和中间轨迹（如果requested）
        """
        # 验证模型输出维度是否与num_actions匹配
        if hasattr(self.model, 'max_actions') and self.model.max_actions != self.num_actions:
            raise ValueError(
                f"模型的max_actions({self.model.max_actions})与扩散模型的num_actions({self.num_actions})不匹配"
            )
            
        # 获取批次大小
        B = obs[0].size(0)
        device = obs[0].device
        
        # T=0时退化成普通DQL，直接使用模型预测
        if self.T == 0:
            # 检查intermediate采样标志
            if return_intermediate:
                raise ValueError("T=0模式不支持中间采样")
            
            # 创建dummy时间步（全为0）
            t = torch.zeros(B, dtype=torch.long, device=device)
            
            # 创建dummy输入logits
            logits_t = torch.zeros(B, self.num_actions, device=device, dtype=torch.float32)
            
            # 计算有效动作数（基于padding_mask）
            num_valid_actions = padding_mask.shape[2] - padding_mask.sum(dim=2).long().squeeze(1)
            
            # 应用padding mask到输入logits
            action_indices = torch.arange(self.num_actions, device=device, dtype=logits_t.dtype)
            mask = action_indices.expand_as(logits_t) >= num_valid_actions.unsqueeze(1)
            logits_t = logits_t.masked_fill(mask, off_value)
            
            # 直接使用模型预测
            pred_logits = self.model(logits_t, t, obs)
            
            # 在pointer模式下，需要调整有效动作数
            model_output_dim = pred_logits.size(-1)
            # 重新计算mask用于最终输出
            action_indices_output = torch.arange(model_output_dim, device=device, dtype=pred_logits.dtype)
            valid_mask = action_indices_output.expand(B, -1) < num_valid_actions.unsqueeze(1)
            
            # 应用mask并计算最终logits
            final_logits = torch.where(valid_mask, pred_logits, torch.full_like(pred_logits, off_value))
            
            return final_logits, None
        
        # T>0时的正常扩散采样过程
        # 预先计算图嵌入，避免在每个扩散步骤中重复计算
        encode_obs_cache = None
        if hasattr(self.model, 'encode_obs'):
            try:
                vp_feat, cur_feat, g_feat = self.model.encode_obs(obs)
                encode_obs_cache = (vp_feat, cur_feat, g_feat)
            except Exception as e:
                warnings.warn(f"无法预计算图嵌入: {e}")
        
        # 计算每个批次样本的有效动作数量
        num_valid_actions = padding_mask.shape[2] - padding_mask.sum(dim=2).long()  # [B, 1]
        num_valid_actions = num_valid_actions.squeeze(1)  # [B]

        # 初始化随机logits
        logits_t = torch.randn(B, self.num_actions, device=device, dtype=torch.float32) * self.init_scale
        
        # 创建掩码并应用
        action_indices = torch.arange(self.num_actions, device=device, dtype=logits_t.dtype)
        mask = action_indices.expand_as(logits_t) >= num_valid_actions.unsqueeze(1)
        logits_t = logits_t.masked_fill(mask, off_value)
        
        # 执行逆向扩散过程
        traj = [logits_t.clone()] if return_intermediate else None
        
        for t in reversed(range(self.T)):
            logits_t = self._p_sample_differentiable(
                logits_t, 
                t, 
                obs, 
                num_valid_actions=num_valid_actions, 
                encode_obs_cache=encode_obs_cache,
                off_value=off_value
            )
            
            if return_intermediate:
                traj.append(logits_t.clone())
        
        # 应用最终mask
        valid_mask = action_indices.expand_as(logits_t) < num_valid_actions.unsqueeze(1)
        final_logits = torch.where(valid_mask, logits_t, torch.full_like(logits_t, off_value))

        return final_logits, traj

    @torch.no_grad()
    def sample(
        self,
        obs,
        padding_mask: torch.Tensor,
        return_intermediate: bool = False,
        off_value: float = -1e6,
    ):
        """执行完整的逆向扩散采样过程，生成最终动作索引。
        
        该函数从纯噪声开始，通过逐步去噪，最终生成干净的动作索引。
        当T=0时，直接使用模型作为普通策略网络，跳过所有扩散步骤。
        
        注意：此方法仅支持确定性采样（贪心选择），如需随机采样请使用sample_differentiable方法。
        
        参数
        ----------
        obs : list
            环境观测信息元组（见去噪网络定义）
        padding_mask : torch.Tensor [B, 1, N]
            标识有效动作(0)和填充位置(1)的掩码
        return_intermediate : bool, 默认False
            是否返回中间采样结果
        off_value : float, 默认-64.0
            用于填充无效动作位置的值
            
        返回值
        ----------
        actions : torch.Tensor [B]
            最终生成的动作索引
        traj : list[torch.Tensor], 可选
            如果return_intermediate为True，返回中间采样轨迹 [t=T, …, 0]
        """
        # 调用核心采样逻辑
        final_logits, traj = self._sample_core(
            obs=obs,
            padding_mask=padding_mask,
            return_intermediate=return_intermediate,
            off_value=off_value,
        )
        
        # 贪心选择动作索引
        final_idx = torch.argmax(final_logits, dim=-1)
        
        # 返回结果
        return (final_idx, traj) if return_intermediate else final_idx

    def sample_differentiable(
        self,
        obs,
        padding_mask: torch.Tensor,
        temperature: float = 1.0,
        return_intermediate: bool = False,
        off_value: float = -1e6,
    ):
        """执行可微分的扩散采样过程，用于训练时的梯度回传。
        
        该函数使用完全可微分的方式实现采样，允许梯度通过采样操作反向传播。
        主要用于actor-critic训练中，使QL损失的梯度能够回传到actor网络。
        
        参数
        ----------
        obs : list
            环境观测信息元组（见去噪网络定义）
        padding_mask : torch.Tensor [B, 1, N]
            标识有效动作(0)和填充位置(1)的掩码
        temperature : float, 默认=1.0
            采样温度参数，控制分布的尖锐程度
        return_intermediate : bool, 默认False
            是否返回中间采样结果
        off_value : float, 默认=-64.0
            用于填充无效动作位置的值
            
        返回值
        ----------
        action_probs : torch.Tensor [B, K]
            动作概率分布（完全可微分）
        traj : list[torch.Tensor], 可选
            如果return_intermediate为True，返回中间采样轨迹 [t=T, …, 0]
        """
        # 调用核心采样逻辑
        final_logits, traj = self._sample_core(
            obs=obs,
            padding_mask=padding_mask,
            return_intermediate=return_intermediate,
            off_value=off_value,
        )
        
        # 使用softmax计算概率分布
        action_probs = F.softmax(final_logits / temperature, dim=-1)
        
        # 返回结果
        return (action_probs, traj) if return_intermediate else action_probs

    def _p_sample_differentiable(
        self,
        logits_t: torch.Tensor,
        t: int,
        obs,
        num_valid_actions: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        encode_obs_cache: Optional[tuple] = None,
        off_value: float = -1e6,  # 修改默认值为更严格的mask
    ) -> torch.Tensor:
        """执行单步可微分逆向扩散，从t时刻到t-1时刻。
        
        与_p_sample不同，这个函数保持完全可微分，不使用torch.no_grad()。
        
        参数
        ----------
        logits_t : torch.Tensor [B, K]
            t时刻的动作logits
        t : int
            当前时间步
        obs : list
            环境观测信息（见去噪网络定义）
        num_valid_actions : torch.Tensor [B], 可选
            每个批次样本的有效动作数量（原始模式）
        action_mask : torch.Tensor [B, num_actions], 可选
            动作有效性掩码（简化模式）
        encode_obs_cache : Optional[tuple], 默认None
            预计算的图嵌入，用于避免重复计算
        off_value : float, 默认-64.0
            用于填充无效动作位置的值
            
        返回值
        ----------
        torch.Tensor
            t-1时刻的预测logits
        """
        # 1) 创建批次时间步张量
        t_tensor = torch.full((logits_t.size(0),), t, dtype=torch.long, device=logits_t.device)  # [B]
        
        # 2) 使用去噪网络预测干净动作logits - 保持可微分
        pred_logits = self.model(logits_t, t_tensor, obs, encode_obs_cache=encode_obs_cache)
        
        # T=0时直接返回预测结果（无扩散步骤）
        if self.T == 0:
            # 应用num_valid_actions mask到预测结果
            if num_valid_actions is not None:
                device = pred_logits.device
                dtype = pred_logits.dtype
                action_indices = torch.arange(pred_logits.size(-1), device=device, dtype=dtype)
                mask = action_indices.expand_as(pred_logits) >= num_valid_actions.unsqueeze(1)
                pred_logits = torch.where(~mask, pred_logits, torch.full_like(pred_logits, off_value))
            return pred_logits
        
        # T>0时的正常扩散步骤
        # 3) 线性插值：logits_{t-1} = (1-βₜ) · logits_t  + βₜ · pred_logits
        beta_t = self.betas[t].view(-1, 1).type_as(logits_t)  # [B, 1]
        logits_tm1 = (1.0 - beta_t) * logits_t + beta_t * pred_logits

        # 4) 根据模式应用不同的掩码策略
        # 原始模式：使用num_valid_actions
        device = logits_tm1.device
        dtype = logits_tm1.dtype
        action_indices = torch.arange(self.num_actions, device=device, dtype=dtype)
        mask = action_indices.expand_as(logits_tm1) >= num_valid_actions.unsqueeze(1)
        logits_tm1 = torch.where(~mask, logits_tm1, torch.full_like(logits_tm1, off_value))

        return logits_tm1
