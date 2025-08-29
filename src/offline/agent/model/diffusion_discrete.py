from __future__ import annotations
from typing import Optional
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedules import (
    linear_action_schedule,
    cosine_action_schedule,
    q_sample_logits,
)
from .categorical_utils import (
    sample_action_indices,
    gumbel_softmax_sample,  # 新增导入
)

class DiscreteDiffusion(nn.Module):
    """离散扩散模型，用于离散动作空间的生成和学习。
    
    该模型实现了离散扩散概率模型(DDPM)，特别适用于离散动作空间。
    模型通过一系列的前向扩散步骤向干净动作添加噪声，然后学习
    逆向扩散过程以从噪声中恢复干净动作。

    支持两种去噪器模式：
    1. 原始pointer-network模式：动态动作空间，使用viewpoint padding mask
    2. 简化固定模式：固定动作空间，使用action mask

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
    use_fixed_actions : bool, 默认 False
        是否使用固定动作空间模式（简化模式）。
    """

    def __init__(
        self,
        num_actions: int,
        model: nn.Module,
        T: int = 20,
        schedule: str = "linear",
        use_fixed_actions: bool = False,
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
        self.use_fixed_actions = use_fixed_actions  # 模式选择
        
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
    # 辅助方法 - 用于简化模式的action mask处理
    # ------------------------------------------------------------------
    def _create_action_mask_from_viewpoints(self, viewpoint_padding_mask: torch.Tensor) -> torch.Tensor:
        """从viewpoint padding_mask创建action_mask（用于简化模式）。
        
        参数：
            viewpoint_padding_mask: [B, 1, max_vps] 其中1表示填充的viewpoint位置
            
        返回：
            action_mask: [B, num_actions] 其中True表示有效动作
        """
        B = viewpoint_padding_mask.size(0)
        max_vps = viewpoint_padding_mask.size(2)
        
        # 确保num_actions能容纳所有viewpoints
        if self.num_actions < max_vps:
            warnings.warn(f"num_actions({self.num_actions}) < viewpoints数({max_vps})")
        
        # 创建action mask
        action_mask = torch.zeros(B, self.num_actions, dtype=torch.bool, device=viewpoint_padding_mask.device)
        valid_viewpoints = ~viewpoint_padding_mask.squeeze(1).bool()  # [B, max_vps]
        
        max_copy = min(max_vps, self.num_actions)
        action_mask[:, :max_copy] = valid_viewpoints[:, :max_copy]
        
        return action_mask

    def _apply_action_mask(self, logits: torch.Tensor, action_mask: torch.Tensor, off_value: float = -64.0) -> torch.Tensor:
        """对logits应用action mask（用于简化模式）。
        
        参数：
            logits: [B, num_actions] 动作logits
            action_mask: [B, num_actions] True表示有效动作
            off_value: 无效动作位置的填充值（默认改为-1e6以彻底屏蔽）
            
        返回：
            masked_logits: [B, num_actions] 应用mask后的logits
        """
        return torch.where(action_mask, logits, torch.full_like(logits, off_value))

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
            # T=0模式只支持pointer模式，不支持fixed_actions模式
            if self.use_fixed_actions:
                raise ValueError("T=0模式不支持use_fixed_actions=True，请使用pointer模式")
            
            # 创建dummy时间步（全为0）
            t = torch.zeros(B, dtype=torch.long, device=action_indices0.device)

            # 创建dummy输入logits（保持与模型架构一致的维度）
            # 即使在pointer模式下，我们也使用num_actions维度作为输入
            logits_t = torch.zeros(B, self.num_actions, device=action_indices0.device, dtype=torch.float32)
            
            # 计算有效动作数（基于padding_mask）
            num_valid_actions = padding_mask.shape[2] - padding_mask.sum(dim=2).long().squeeze(1)
            
            # 应用padding mask到输入logits（基于实际的动作空间）
            action_indices = torch.arange(self.num_actions, device=logits_t.device, dtype=logits_t.dtype)
            mask = action_indices.expand_as(logits_t) >= num_valid_actions.unsqueeze(1)
            logits_t = logits_t.masked_fill(mask, -64.0)
            
            # 直接使用模型预测
            pred_logits = self.model(logits_t, t, obs)
            
            # 计算损失 - 注意pred_logits的维度可能与输入不同（pointer模式输出6维）
            loss = F.cross_entropy(pred_logits, action_indices0, reduction="mean")
            
            # 损失字典（T=0时的特殊标记）
            loss_dict = {
                'diff/t_sample': torch.tensor(0.0, device=action_indices0.device),  # T=0模式
                'diff/entropy_xt': torch.tensor(0.0, device=action_indices0.device),  # 无扩散熵
                'diff/loss_ce': loss.item(),  # CE损失
                'diff/loss_consistency': 0.0,  # 无一致性损失
            }
            
            return loss, loss_dict
        
        # T>0时的正常扩散过程
        # 随机采样时间步t
        t = torch.randint(0, self.T, (B,), device=action_indices0.device)
        
        if self.use_fixed_actions:
            # 简化模式：使用固定动作空间和action mask
            # 从obs中提取viewpoint_padding_mask来创建正确的action mask
            if len(obs) >= 5:
                viewpoint_padding_mask = obs[4]  # [B, 1, max_vps]
                action_mask = self._create_action_mask_from_viewpoints(viewpoint_padding_mask)
            else:
                # 后备方案：假设所有动作都有效
                action_mask = torch.ones(B, self.num_actions, dtype=torch.bool, device=action_indices0.device)
            
            # 1. 将动作索引转换为one-hot logits (在num_actions空间中)
            logits0 = torch.full((B, self.num_actions), -64.0, device=action_indices0.device, dtype=torch.float32)
            
            # 确保action_indices0在有效范围内
            valid_indices = action_indices0 < self.num_actions
            if not valid_indices.all():
                warnings.warn(f"部分动作索引超出范围: {action_indices0[~valid_indices]}")
                action_indices0 = action_indices0.clamp(0, self.num_actions - 1)
            
            logits0.scatter_(1, action_indices0.unsqueeze(1), 0.0)  # 在真实动作位置设为0
            
            # 应用action mask
            logits0 = self._apply_action_mask(logits0, action_mask)
            
            # 2. 前向加噪
            beta = self.betas[t].view(-1, 1)  # [B, 1]
            noise = torch.randn_like(logits0) * self.init_scale
            logits_t = (1.0 - beta) * logits0 + beta * noise
            
            # 重新应用mask确保无效动作保持-64
            logits_t = self._apply_action_mask(logits_t, action_mask)
            
            # 3. 去噪预测
            pred_logits = self.model(logits_t, t, obs)
            
            # 验证预测输出的维度
            if pred_logits.shape[-1] != self.num_actions:
                raise RuntimeError(
                    f"模型输出维度({pred_logits.shape[-1]})与期望的num_actions({self.num_actions})不匹配"
                )
            
            # 4. 应用mask到预测结果
            pred_logits = self._apply_action_mask(pred_logits, action_mask)
        else:
            # 原始模式：使用动态动作空间
            # 1. 前向加噪 (logits-space)
            logits_t = q_sample_logits(action_indices0, t, self.betas, padding_mask, init_scale=self.init_scale)

            # 2. 去噪预测
            pred_logits = self.model(logits_t, t, obs)

        # 3. 计算两步损失：CE损失 + 一致性损失
        # 主要损失：交叉熵 → 等价 KL(one-hot‖softmax(pred))
        loss_ce = F.cross_entropy(pred_logits, action_indices0, reduction="mean")
        
        # 一致性损失：teacher forcing
        loss_consistency = 0.0
        consistency_weight = 0.0  # 一致性损失权重
        
        if self.T > 1:  # 只有多步扩散时才计算一致性损失
            with torch.no_grad():
                # 计算t-1时刻的teacher logits
                t_prev = torch.clamp(t - 1, min=0)  # 防止t=0时越界
                
                if self.use_fixed_actions:
                    # 简化模式：重新计算t-1时刻的logits
                    beta_prev = self.betas[t_prev].view(-1, 1)  # [B, 1]
                    noise_prev = torch.randn_like(logits0) * self.init_scale
                    logits_tm1_teacher = (1.0 - beta_prev) * logits0 + beta_prev * noise_prev
                    logits_tm1_teacher = self._apply_action_mask(logits_tm1_teacher, action_mask)
                else:
                    # 原始模式：重新计算t-1时刻的logits
                    logits_tm1_teacher = q_sample_logits(action_indices0, t_prev, self.betas, padding_mask, init_scale=self.init_scale)
                
                # teacher预测
                logits_tm1_pred = self.model(logits_tm1_teacher, t_prev, obs)
            
            # KL散度一致性损失
            log_pred = F.log_softmax(pred_logits, dim=-1)
            soft_teacher = F.softmax(logits_tm1_pred, dim=-1)
            loss_consistency = F.kl_div(log_pred, soft_teacher, reduction="batchmean")
        
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
        # 1) 创建批次时间步张量
        t_tensor = torch.full((logits_t.size(0),), t, dtype=torch.long, device=logits_t.device)  # [B]
        
        # 2) 使用去噪网络预测干净动作logits
        pred_logits = self.model(logits_t, t_tensor, obs, encode_obs_cache=encode_obs_cache)
        
        # T=0时直接返回预测结果（无扩散步骤）
        if self.T == 0:
            # T=0模式只支持pointer模式
            if self.use_fixed_actions:
                raise ValueError("T=0模式不支持use_fixed_actions=True，请使用pointer模式")
            
            # 应用num_valid_actions mask到预测结果
            if num_valid_actions is not None:
                device = pred_logits.device
                dtype = pred_logits.dtype
                action_indices = torch.arange(pred_logits.size(-1), device=device, dtype=dtype)
                mask = action_indices.expand_as(pred_logits) >= num_valid_actions.unsqueeze(1)
                pred_logits = pred_logits.masked_fill(mask, off_value)
            return pred_logits
        
        # T>0时的正常扩散步骤
        # 3) 线性插值：logits_{t-1} = (1-βₜ) · logits_t  + βₜ · pred_logits
        beta_t = self.betas[t].view(-1, 1).type_as(logits_t)  # [B, 1]
        logits_tm1 = (1.0 - beta_t) * logits_t + beta_t * pred_logits

        # 4) 根据模式应用不同的掩码策略
        if self.use_fixed_actions and action_mask is not None:
            # 简化模式：使用action mask
            logits_tm1 = self._apply_action_mask(logits_tm1, action_mask, off_value)
        elif num_valid_actions is not None:
            # 原始模式：使用num_valid_actions
            device = logits_tm1.device
            dtype = logits_tm1.dtype
            action_indices = torch.arange(self.num_actions, device=device, dtype=dtype)
            mask = action_indices.expand_as(logits_tm1) >= num_valid_actions.unsqueeze(1)
            logits_tm1 = logits_tm1.masked_fill(mask, off_value)

        return logits_tm1

    @torch.no_grad()
    def sample(
        self,
        obs,
        padding_mask: torch.Tensor,
        return_intermediate: bool = False,
        off_value: float = -64.0,  # 修改默认值为更严格的mask
    ):
        """执行完整的逆向扩散采样过程，生成最终动作索引。
        
        该函数从纯噪声开始，通过逐步去噪，最终生成干净的动作索引。
        当T=0时，直接使用模型作为普通策略网络，跳过所有扩散步骤。
        支持两种模式：
        1. 原始模式：使用动态动作空间和num_valid_actions
        2. 简化模式：使用固定动作空间和action_mask
        
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
            # T=0模式只支持pointer模式
            if self.use_fixed_actions:
                raise ValueError("T=0模式不支持use_fixed_actions=True，请使用pointer模式")
                
            # 检查intermediate采样标志
            assert not return_intermediate, "T=0模式不支持中间采样"
            
            # 创建dummy时间步（全为0）
            t = torch.zeros(B, dtype=torch.long, device=device)
            
            # 创建dummy输入logits（保持与模型架构一致的维度）
            # 即使在pointer模式下，我们也使用num_actions维度作为输入
            logits_t = torch.zeros(B, self.num_actions, device=device, dtype=torch.float32)
            
            # 计算有效动作数（基于padding_mask）
            num_valid_actions = padding_mask.shape[2] - padding_mask.sum(dim=2).long().squeeze(1)
            
            # 应用padding mask到输入logits（基于实际的动作空间）
            action_indices = torch.arange(self.num_actions, device=device, dtype=logits_t.dtype)
            mask = action_indices.expand_as(logits_t) >= num_valid_actions.unsqueeze(1)
            logits_t = logits_t.masked_fill(mask, off_value)
            
            # 直接使用模型预测
            pred_logits = self.model(logits_t, t, obs)
            
            # 在pointer模式下，模型输出维度可能与输入不同
            # 需要计算正确的有效动作数用于采样
            if hasattr(self.model, 'use_pointer_network') and self.model.use_pointer_network:
                # Pointer模式：有效动作数应该是模型输出维度
                model_output_dim = pred_logits.size(-1)
                num_valid_actions_for_sampling = torch.full_like(num_valid_actions, model_output_dim)
            else:
                # 非pointer模式：使用原始的有效动作数
                num_valid_actions_for_sampling = num_valid_actions
            
            # 采样最终动作（贪心选择）
            final_idx = sample_action_indices(
                action_logits=pred_logits, 
                num_valid_actions=num_valid_actions_for_sampling, 
                stochastic=False,  # 固定为贪心选择
                off_value=off_value
            )
            
            # 返回结果
            return final_idx
        
        # T>0时的正常扩散采样过程
        # 预先计算图嵌入，避免在每个扩散步骤中重复计算
        encode_obs_cache = None
        if hasattr(self.model, 'encode_obs'):
            try:
                vp_feat, cur_feat, g_feat = self.model.encode_obs(obs)
                encode_obs_cache = (vp_feat, cur_feat, g_feat)
            except Exception as e:
                warnings.warn(f"无法预计算图嵌入: {e}")
        
        if self.use_fixed_actions:
            # === 简化模式：使用action mask ===
            # 从obs中提取viewpoint_padding_mask来创建action mask
            if len(obs) >= 5:
                viewpoint_padding_mask = obs[4]  # [B, 1, max_vps]
                action_mask = self._create_action_mask_from_viewpoints(viewpoint_padding_mask)
            else:
                # 后备方案：假设所有动作都有效
                action_mask = torch.ones(B, self.num_actions, dtype=torch.bool, device=device)
            
            # 初始化随机logits
            logits_t = torch.randn(B, self.num_actions, device=device) * self.init_scale
            logits_t = self._apply_action_mask(logits_t, action_mask, off_value)
            
            # 执行逆向扩散过程
            traj = [logits_t.clone()] if return_intermediate else None
            
            for t in reversed(range(self.T)):
                logits_t = self._p_sample(
                    logits_t, 
                    t, 
                    obs, 
                    action_mask=action_mask,
                    encode_obs_cache=encode_obs_cache,
                    off_value=off_value
                )
                
                if return_intermediate:
                    traj.append(logits_t.clone())
            
            # 贪心选择最大概率的有效动作
            masked_logits = self._apply_action_mask(logits_t, action_mask, off_value)
            final_idx = torch.argmax(masked_logits, dim=-1)
        else:
            # === 原始模式：使用num_valid_actions ===
            # 计算每个批次样本的有效动作数量
            num_valid_actions = padding_mask.shape[2] - padding_mask.sum(dim=2).long()  # [B, 1]
            num_valid_actions = num_valid_actions.squeeze(1)  # [B]

            # 初始化随机logits
            logits_t = torch.randn(B, self.num_actions, device=device) * self.init_scale
            
            # 创建掩码并应用
            device_type = logits_t.device
            dtype_type = logits_t.dtype
            action_indices = torch.arange(self.num_actions, device=device_type, dtype=dtype_type)
            mask = action_indices.expand_as(logits_t) >= num_valid_actions.unsqueeze(1)
            logits_t = logits_t.masked_fill(mask, off_value)
            
            # 执行逆向扩散过程
            traj = [logits_t.clone()] if return_intermediate else None
            
            for t in reversed(range(self.T)):
                logits_t = self._p_sample(
                    logits_t, 
                    t, 
                    obs, 
                    num_valid_actions=num_valid_actions, 
                    encode_obs_cache=encode_obs_cache,
                    off_value=off_value
                )
                
                if return_intermediate:
                    traj.append(logits_t.clone())
            
            # 使用原有的采样方法（贪心选择）
            final_idx = sample_action_indices(
                action_logits=logits_t, 
                num_valid_actions=num_valid_actions, 
                stochastic=False,  # 固定为贪心选择
                off_value=off_value
            )

        # 返回结果
        return (final_idx, traj) if return_intermediate else final_idx

    def sample_differentiable(
        self,
        obs,
        padding_mask: torch.Tensor,
        temperature: float = 1.0,
        off_value: float = -64.0,  # 修改默认值为更严格的mask
        use_gumbel_softmax: bool = False,
        gumbel_hard: bool = False,
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
        off_value : float, 默认=-64.0
            用于填充无效动作位置的值
        use_gumbel_softmax : bool, 默认=False
            是否使用Gumbel-Softmax采样而非标准softmax
        gumbel_hard : bool, 默认=False
            当use_gumbel_softmax=True时，是否使用直通估计器
            True时前向传播返回one-hot，反向传播使用soft概率，减少train-test mismatch
            
        返回值
        ----------
        action_probs : torch.Tensor [B, K]
            动作概率分布（完全可微分）
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
            # T=0模式只支持pointer模式
            if self.use_fixed_actions:
                raise ValueError("T=0模式不支持use_fixed_actions=True，请使用pointer模式")
                
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
            if hasattr(self.model, 'use_pointer_network') and self.model.use_pointer_network:
                model_output_dim = pred_logits.size(-1)
                # 重新计算mask用于最终输出
                action_indices_output = torch.arange(model_output_dim, device=device, dtype=pred_logits.dtype)
                valid_mask = action_indices_output.expand(B, -1) < num_valid_actions.unsqueeze(1)
                
                # 应用mask并计算概率分布
                masked_logits = torch.where(valid_mask, pred_logits, torch.full_like(pred_logits, off_value))
                
                if use_gumbel_softmax:
                    # 使用Gumbel-Softmax（pointer模式）
                    action_probs = gumbel_softmax_sample(
                        masked_logits, 
                        num_valid_actions, 
                        temperature=temperature, 
                        hard=gumbel_hard,
                        off_value=off_value
                    )
                else:
                    # 使用标准softmax
                    action_probs = F.softmax(masked_logits / temperature, dim=-1)
            else:
                # 非pointer模式：使用原始的有效动作数
                valid_mask = action_indices.expand_as(pred_logits) < num_valid_actions.unsqueeze(1)
                masked_logits = torch.where(valid_mask, pred_logits, torch.full_like(pred_logits, off_value))
                
                if use_gumbel_softmax:
                    # 使用Gumbel-Softmax（非pointer模式）
                    action_probs = gumbel_softmax_sample(
                        masked_logits, 
                        num_valid_actions, 
                        temperature=temperature, 
                        hard=gumbel_hard,
                        off_value=off_value
                    )
                else:
                    # 使用标准softmax
                    action_probs = F.softmax(masked_logits / temperature, dim=-1)
            
            return action_probs
        
        # T>0时的正常扩散采样过程
        # 预先计算图嵌入，避免在每个扩散步骤中重复计算
        encode_obs_cache = None
        if hasattr(self.model, 'encode_obs'):
            try:
                vp_feat, cur_feat, g_feat = self.model.encode_obs(obs)
                encode_obs_cache = (vp_feat, cur_feat, g_feat)
            except Exception as e:
                warnings.warn(f"无法预计算图嵌入: {e}")
        
        if self.use_fixed_actions:
            # === 简化模式：使用action mask ===
            # 从obs中提取viewpoint_padding_mask来创建action mask
            if len(obs) >= 5:
                viewpoint_padding_mask = obs[4]  # [B, 1, max_vps]
                action_mask = self._create_action_mask_from_viewpoints(viewpoint_padding_mask)
            else:
                # 后备方案：假设所有动作都有效
                action_mask = torch.ones(B, self.num_actions, dtype=torch.bool, device=device)
            
            # 初始化随机logits - 修复梯度流问题
            # 确保初始噪声变量具有正确的梯度设置
            logits_t = torch.randn(B, self.num_actions, device=device, dtype=torch.float32, requires_grad=True) * self.init_scale
            logits_t = self._apply_action_mask(logits_t, action_mask, off_value)
            
            # 执行逆向扩散过程 - 使用可微分版本
            for t in reversed(range(self.T)):
                logits_t = self._p_sample_differentiable(
                    logits_t, 
                    t, 
                    obs, 
                    action_mask=action_mask,
                    encode_obs_cache=encode_obs_cache,
                    off_value=off_value
                )
            
            # 应用最终mask并计算概率分布
            final_logits = self._apply_action_mask(logits_t, action_mask, off_value)
            
            if use_gumbel_softmax:
                # 使用Gumbel-Softmax，需要计算有效动作数
                num_valid_actions = action_mask.sum(dim=-1)  # [B]
                action_probs = gumbel_softmax_sample(
                    final_logits, 
                    num_valid_actions, 
                    temperature=temperature, 
                    hard=gumbel_hard,
                    off_value=off_value
                )
            else:
                # 使用标准softmax
                action_probs = F.softmax(final_logits / temperature, dim=-1)
            
        else:
            # === 原始模式：使用num_valid_actions ===
            # 计算每个批次样本的有效动作数量
            num_valid_actions = padding_mask.shape[2] - padding_mask.sum(dim=2).long()  # [B, 1]
            num_valid_actions = num_valid_actions.squeeze(1)  # [B]

            # 初始化随机logits - 修复梯度流问题
            # 确保初始噪声变量具有正确的梯度设置
            logits_t = torch.randn(B, self.num_actions, device=device, dtype=torch.float32, requires_grad=True) * self.init_scale
            
            # 创建掩码并应用
            action_indices = torch.arange(self.num_actions, device=device, dtype=logits_t.dtype)
            mask = action_indices.expand_as(logits_t) >= num_valid_actions.unsqueeze(1)
            logits_t = logits_t.masked_fill(mask, off_value)
            
            # 执行逆向扩散过程 - 使用可微分版本
            for t in reversed(range(self.T)):
                logits_t = self._p_sample_differentiable(
                    logits_t, 
                    t, 
                    obs, 
                    num_valid_actions=num_valid_actions, 
                    encode_obs_cache=encode_obs_cache,
                    off_value=off_value
                )
            
            # 应用最终mask并计算概率分布
            valid_mask = action_indices.expand_as(logits_t) < num_valid_actions.unsqueeze(1)
            masked_logits = torch.where(valid_mask, logits_t, torch.full_like(logits_t, off_value))
            
            if use_gumbel_softmax:
                # 使用Gumbel-Softmax
                action_probs = gumbel_softmax_sample(
                    masked_logits, 
                    num_valid_actions, 
                    temperature=temperature, 
                    hard=gumbel_hard,
                    off_value=off_value
                )
            else:
                # 使用标准softmax
                action_probs = F.softmax(masked_logits / temperature, dim=-1)

        return action_probs

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
            # T=0模式只支持pointer模式
            if self.use_fixed_actions:
                raise ValueError("T=0模式不支持use_fixed_actions=True，请使用pointer模式")
            
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
        if self.use_fixed_actions and action_mask is not None:
            # 简化模式：使用action mask
            logits_tm1 = self._apply_action_mask(logits_tm1, action_mask, off_value)
        elif num_valid_actions is not None:
            # 原始模式：使用num_valid_actions
            device = logits_tm1.device
            dtype = logits_tm1.dtype
            action_indices = torch.arange(self.num_actions, device=device, dtype=dtype)
            mask = action_indices.expand_as(logits_tm1) >= num_valid_actions.unsqueeze(1)
            logits_tm1 = torch.where(~mask, logits_tm1, torch.full_like(logits_tm1, off_value))

        return logits_tm1
