"""
unified_graph_denoiser.py
-------------------------------------------------------------
统一的图去噪器，支持两种扩散方式：
1. Pointer Network模式：使用pointer attention选择动作
2. 简化模式：直接生成固定大小的动作logits

通过参数配置来选择不同的diffusion方式，代码复用度更高。
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .sgformer import PolicyNet


# -----------------------------------------------------------------------------
# 扩散步骤t的位置嵌入 (复用原始实现)
# -----------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """扩散时间步的正弦位置嵌入。
    
    使用不同频率的正弦和余弦函数创建时间嵌入，类似于Transformer位置编码。
    
    参数：
        dim: 嵌入维度（必须为偶数）。
    """
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"时间嵌入维度必须为偶数，得到{dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """将时间步转换为嵌入向量。
        
        参数：
            t: 时间步张量 [B]
            
        返回：
            时间嵌入 [B, dim]
        """
        half = self.dim // 2
        emb_fac = math.log(10000.0) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -emb_fac)
        emb = t[:, None] * emb[None, :]  # [B, half]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)  # [B, dim]


# -----------------------------------------------------------------------------
# 统一的图去噪器
# -----------------------------------------------------------------------------

class GraphActionDenoiser(nn.Module):
    """统一的图动作去噪器，支持两种扩散方式。
    
    该模型可以通过配置参数在两种模式间切换：
    1. Pointer Network模式：使用FiLM门控和pointer attention（原始方式）
    2. 简化模式：直接的特征融合和MLP输出（简化方式）

    参数：
        node_dim: 输入图中每个节点特征的维度。
        embed_dim: 图编码器使用的隐藏大小。
        max_actions: 最大动作空间大小（用于生成固定大小的logits）。
        t_dim: 时间步t的嵌入维度，默认16。
        T: 扩散总时间步数，用于归一化时间步，默认20。
        logits_scale: logits压缩因子，用于tanh(logits/scale)中，默认8。
        use_pointer_network: 是否使用pointer network模式，默认True。
        action_proj_dim: pointer模式下压缩噪声对数似然的维度，默认64。
        temperature: 输出logits的温度参数，默认5.0（pointer）或1.0（简化）。
    """

    def __init__(
        self,
        node_dim: int,
        embed_dim: int,
        max_actions: int,
        t_dim: int = 16,
        T: int = 20,
        logits_scale: float = 8.0,
        use_pointer_network: bool = True,
        action_proj_dim: int = 64,
        temperature: Optional[float] = None,
    ):
        super().__init__()
        
        # 验证输入参数
        if max_actions <= 0:
            raise ValueError(f"max_actions必须为正数，得到{max_actions}")
        if t_dim <= 0 or t_dim % 2 != 0:
            raise ValueError(f"t_dim必须为正偶数，得到{t_dim}")
        if T < 0:  # 修改：允许T=0，退化成普通策略网络
            raise ValueError(f"T必须为非负数，得到{T}")
            
        self.max_actions = max_actions
        self.embed_dim = embed_dim
        self.t_dim = t_dim
        self.T = T
        self.logits_scale = logits_scale
        self.use_pointer_network = use_pointer_network
        self.action_proj_dim = action_proj_dim
        
        # 根据模式设置默认温度
        if temperature is None:
            self.temperature = 1.0 if use_pointer_network else 1.0
        else:
            self.temperature = temperature

        # 初始化图编码器（两种模式都需要）
        self.policy = PolicyNet(node_dim, embed_dim)

        # 时间步嵌入网络（两种模式都需要）
        self.time_mlp = self._build_time_mlp()

        if use_pointer_network:
            # Pointer Network模式的特定组件
            self.action_proj = nn.Linear(max_actions, action_proj_dim, bias=False)
            self.time_mix = nn.Linear(t_dim, embed_dim, bias=False)
            self.gamma = nn.Parameter(torch.zeros(1))  # FiLM门控参数
        else:
            # 简化模式的特定组件
            self.action_proj = nn.Sequential(
                nn.Linear(max_actions, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim // 2),
            )
            
            # 特征融合网络
            fusion_input_dim = embed_dim * 2 + embed_dim // 2 + t_dim
            self.feature_fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
            )
            
            # 输出层
            self.output_head = nn.Linear(embed_dim, max_actions)
    
    def _build_time_mlp(self) -> nn.Sequential:
        """构建时间嵌入MLP。
        
        返回：
            用于时间嵌入的Sequential模块。
        """
        return nn.Sequential(
            SinusoidalPosEmb(self.t_dim),
            nn.Linear(self.t_dim, self.t_dim * 4),
            nn.Mish(),
            nn.Linear(self.t_dim * 4, self.t_dim),
        )

    def _encode_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """编码扩散时间步。
        
        参数：
            t: 时间步索引 [B]
            
        返回：
            时间嵌入 [B, t_dim]
        """
        # T=0时的特殊处理
        if self.T == 0:
            # 当T=0时，返回零时间嵌入（表示无扩散步骤）
            return torch.zeros(t.size(0), self.t_dim, device=t.device, dtype=torch.float32)
        
        # 使用self.T将时间步归一化到[0, 1]区间
        t_norm = t.float() / self.T
        return self.time_mlp(t_norm)  # [B, t_dim]

    def encode_obs(self, obs):
        """计算并返回图状态嵌入，用于避免重复计算。
        
        参数：
            obs: 环境观测列表，包含：
                - node_inputs: 节点特征 [B, N, node_dim]
                - node_padding_mask: 有效节点的掩码 [B, 1, N] (0=有效，1=填充)
                - current_index: 当前节点索引 [B, 1, 1]
                - viewpoints: 视点索引 [B, max_num_vps, 1]
                - viewpoint_padding_mask: 视点掩码 [B, 1, max_num_vps] (0=有效，1=填充)
                - adj_list: 邻接列表 [B, N, K]
                
        返回：
            - viewpoint_feature: 视点嵌入 [B, max_num_vps, embed_dim]
            - current_node_feature: 当前节点嵌入 [B, 1, embed_dim]
            - global_node_feature: 全局节点嵌入 [B, 1, embed_dim]
        """
        node_inputs, node_padding_mask, current_index, viewpoints, _, adj_list = obs
        
        # 获取图状态嵌入
        node_feature = self.policy.encode_graph(node_inputs, adj_list)
        current_node_feature, global_node_feature = self.policy.decode_state(
            node_feature, current_index, node_padding_mask)
        viewpoint_feature = torch.gather(node_feature, 1,
                                           viewpoints.repeat(1, 1, self.embed_dim).long())
        return viewpoint_feature, current_node_feature, global_node_feature

    def forward(
        self,
        x_t_action_logits: torch.Tensor,  # [B, max_actions] t步的噪声对数似然
        t: torch.Tensor,                  # [B] 时间步索引
        obs: list,                        # 环境观测元组
        encode_obs_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,  # 可选: 缓存的观测编码
    ) -> torch.Tensor:                    # 返回 [B, max_actions] 干净对数似然
        """在给定扩散步骤中，从噪声对数似然预测干净动作对数似然。
        
        参数：
            x_t_action_logits: t扩散步的噪声动作对数似然，形状 [B, max_actions]
            t: 扩散时间步索引，形状 [B]
            obs: 环境观测列表
            encode_obs_cache: 预计算的观测编码（可选）
                
        返回：
            预测的干净动作对数似然，形状 [B, max_actions]
        """
        # ---- Feature 0: 图状态嵌入 ----
        if encode_obs_cache is not None:
            vp_feat, cur_feat, g_feat = encode_obs_cache
        else:
            vp_feat, cur_feat, g_feat = self.encode_obs(obs)
        
        # ---- Feature 1：噪声动作编码 ----
        compressed_logits = torch.tanh(x_t_action_logits / self.logits_scale)
        act_feat = self.action_proj(compressed_logits)  # 两种模式维度不同
        
        # ---- Feature 2：时间步编码 ----
        time_emb = self._encode_timestep(t)  # [B, t_dim]
        
        if self.use_pointer_network:
            # === Pointer Network模式 ===
            # 融合当前节点和全局特征
            cur_feat = self.policy.current_embedding(torch.cat((g_feat, cur_feat), dim=-1))  # [B, 1, E]

            if self.T > 0:
                # 时间信息投影并加到query向量
                time_proj = self.time_mix(time_emb).unsqueeze(1)  # [B, 1, E]
                cur_feat = cur_feat + time_proj
                
                # FiLM门控：用动作特征调制viewpoint特征
                # 确保维度匹配：将action_proj_dim扩展到embed_dim
                if act_feat.shape[-1] != self.embed_dim:
                    # 需要广播到正确维度
                    act_feat_expanded = act_feat.unsqueeze(-1).expand(-1, -1, self.embed_dim // self.action_proj_dim).contiguous()
                    act_feat_expanded = act_feat_expanded.view(act_feat.shape[0], self.embed_dim)
                else:
                    act_feat_expanded = act_feat
                    
                gate = (1 + self.gamma * act_feat_expanded.unsqueeze(1))  # [B, 1, E]
                vp_feat = vp_feat * gate
                
                # Pointer attention
                viewpoint_padding_mask = obs[4]  # [B, 1, K]
                logits = self.policy.pointer(cur_feat, vp_feat, viewpoint_padding_mask)  # [B, 1, K]
                return logits.squeeze(1) / self.temperature  # [B, K]
            else:
                # Pointer attention
                viewpoint_padding_mask = obs[4]  # [B, 1, K]
                logits = self.policy.pointer(cur_feat, vp_feat, viewpoint_padding_mask)  # [B, 1, K]
                return logits.squeeze(1) / self.temperature  # [B, K]
        
        else:
            # === 简化模式 ===
            # 融合当前节点和全局特征
            state_feat = self.policy.current_embedding(torch.cat((g_feat, cur_feat), dim=-1))  # [B, 1, E]
            state_feat = state_feat.squeeze(1)  # [B, E]
            
            # 特征拼接融合
            fused_input = torch.cat([
                state_feat,           # [B, E]
                g_feat.squeeze(1),    # [B, E] 
                act_feat,             # [B, E//2]
                time_emb              # [B, t_dim]
            ], dim=-1)  # [B, E*2 + E//2 + t_dim]
            
            # 通过融合网络
            fused_feat = self.feature_fusion(fused_input)  # [B, E]
            
            # 生成输出logits
            output_logits = self.output_head(fused_feat)  # [B, max_actions]
            
            return output_logits / self.temperature


# -----------------------------------------------------------------------------
# 便捷的工厂函数
# -----------------------------------------------------------------------------

def create_pointer_denoiser(
    node_dim: int,
    embed_dim: int,
    max_actions: int,
    **kwargs
) -> GraphActionDenoiser:
    """创建Pointer Network模式的去噪器。
    
    这是原始复杂模式，使用FiLM门控和pointer attention。
    """
    return GraphActionDenoiser(
        node_dim=node_dim,
        embed_dim=embed_dim,
        max_actions=max_actions,
        use_pointer_network=True,
        **kwargs
    )


def create_simple_denoiser(
    node_dim: int,
    embed_dim: int,
    max_actions: int,
    **kwargs
) -> GraphActionDenoiser:
    """创建简化模式的去噪器。
    
    这是简化模式，使用直接特征融合和MLP输出。
    """
    return GraphActionDenoiser(
        node_dim=node_dim,
        embed_dim=embed_dim,
        max_actions=max_actions,
        use_pointer_network=False,
        **kwargs
    )