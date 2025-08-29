import os
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from dataclasses import asdict, dataclass
from .model.sgformer import PolicyNet as Actor
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    seed: int = 0  
    save_freq: int = int(1e2)  # 评估频率（步数）
    max_timesteps: int = int(3e5)  # 运行环境的最大时间步数
    checkpoints_path: Optional[str] = None  # 保存路径
    load_model_path: Optional[str] = None  # 模型加载路径（新增）
    
    # Buffer
    buffer_size: int = 1_000_000  # 回放缓冲区大小
    
    # Optimization
    learning_rate: float = 3e-4  # 使用统一学习率

    batch_size: int = 256  # 所有网络的批量大小
    mini_batch_size: int = 256  # BC的mini-batch大小
    clip_grad_norm: Optional[float] = 1.0  # 默认开启梯度裁剪
    
    # Wandb日志
    project: str = "BC-Graph"
    group: str = "BC"
    name: str = "BC"
    
    # 图相关参数
    node_dim: int = None      # 节点特征维度
    num_nodes: int = None     # 图中节点数量
    edge_dim: int = None      # 边特征维度
    gnn_hidden_dim: int = 128 # 图神经网络隐藏层维度
    num_gnn_layers: int = 3   # 图神经网络层数

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class DDPBC:
    """
    基于DDP的行为克隆(BC)算法实现 - 图结构特化版本
    纯监督学习方法，通过模仿专家行为来训练策略
    """
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        clip_grad_norm: float = 1.0,
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer

        self.clip_grad_norm = clip_grad_norm
        self.total_it = 0
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        # 保存初始学习率，用于学习率衰减
        self.initial_actor_lr = self._get_lr(self.actor_optimizer)

    def _get_lr(self, optimizer):
        """获取优化器的当前学习率"""
        for param_group in optimizer.param_groups:
            return param_group['lr']
        return None
    
    def get_current_lr(self):
        """获取当前学习率信息，用于日志记录"""
        return {
            "actor_lr": self._get_lr(self.actor_optimizer)
        }

    def train(self, batch: Dict[str, torch.Tensor], mini_batch: int = None) -> Dict[str, float]:
        """
        执行一步行为克隆训练更新，仅更新actor网络
        
        参数:
            batch: 训练数据批次（只需要状态和动作）
            mini_batch: mini-batch大小，如果为None则使用整个batch
        """
        self.total_it += 1
        log_dict = {}

        batch_size = batch['actions'].shape[0]
        
        # 优化微批处理逻辑
        if mini_batch is None or mini_batch >= batch_size:
            slices = [(0, batch_size)]
        else:
            slices = [(i, min(i + mini_batch, batch_size)) for i in range(0, batch_size, mini_batch)]
        
        # 只清空actor的梯度
        self.actor_optimizer.zero_grad(set_to_none=True)
        
        total_bc_loss = 0.0
        
        # 对每个mini-batch进行前向传播和反向传播
        for beg, end in slices:
            mb_size = end - beg
            
            # 提取mini-batch数据
            mb_current_observation = [
                batch['node_inputs'][beg:end], 
                batch['node_padding_mask'][beg:end], 
                batch['current_index'][beg:end], 
                batch['viewpoints'][beg:end], 
                batch['viewpoint_padding_mask'][beg:end], 
                batch['adj_list'][beg:end]
            ]
            
            mb_actions = batch['actions'][beg:end]
            
            # ---------- 只更新Actor网络（行为克隆） ----------
            action_logits = self.actor(mb_current_observation)
            
            # 计算交叉熵损失（行为克隆损失）
            bc_loss = F.cross_entropy(action_logits, mb_actions)
            total_bc_loss += bc_loss.item() * mb_size / batch_size
            
            # 反向传播，但不立即更新参数
            bc_loss.backward()
        
        # 所有mini-batch处理完毕后，进行梯度裁剪并更新参数
        # 梯度裁剪
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        
        # 更新actor参数
        self.actor_optimizer.step()
        
        # 记录训练信息
        log_dict["bc_loss"] = total_bc_loss
        
        return log_dict

    def update_learning_rate(self, progress_remaining):
        """
        使用线性衰减策略更新学习率
        
        参数:
            progress_remaining: 剩余训练进度比例 (0.0 到 1.0)
        """
        # 确保进度值合法
        progress_remaining = np.clip(progress_remaining, 0.00001, 1.0)
        
        # 线性学习率衰减
        new_actor_lr = self.initial_actor_lr * progress_remaining
        
        # 更新actor学习率
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_actor_lr
        
        # 返回当前学习率信息，用于记录
        return {
            "actor_lr": new_actor_lr,
            "lr_factor": progress_remaining
        }

    def update_hyperparameters(self, progress_remaining):
        """
        统一的超参数更新函数，更新学习率
        
        参数:
            progress_remaining: 剩余训练进度比例 (0.0 到 1.0)
        
        返回:
            包含所有超参数更新信息的字典
        """
        # 更新学习率
        lr_info = self.update_learning_rate(progress_remaining)
        
        # 合并所有更新信息
        update_info = {}
        
        # 添加学习率信息
        for k, v in lr_info.items():
            update_info[f"lr/{k}"] = v
        
        return update_info

    @torch.no_grad()
    def select_action(self, observation) -> torch.Tensor:
        """
        选择动作的接口函数
        """
        processed_observation = []
        for i, obs in enumerate(observation):
            if isinstance(obs, np.ndarray):
                processed_observation.append(torch.from_numpy(obs).to(self.device))
            else:
                processed_observation.append(obs.to(self.device))
        
        # 处理单个样本的情况（添加batch维度）
        for i, obs in enumerate(processed_observation):
            if obs.dim() == 2 and i == 0:  # node_inputs
                processed_observation[i] = obs.unsqueeze(0)
            elif obs.dim() == 1 and i in [1, 4]:  # node_padding_mask, viewpoint_padding_mask
                processed_observation[i] = obs.unsqueeze(0).unsqueeze(0)
            elif obs.dim() == 0 and i == 2:  # current_index
                processed_observation[i] = obs.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif obs.dim() == 1 and i == 3:  # viewpoints
                processed_observation[i] = obs.unsqueeze(0).unsqueeze(-1)
            elif obs.dim() == 2 and i == 5:  # adj_list
                processed_observation[i] = obs.unsqueeze(0)
        
        action_index = self.actor.predict(processed_observation)
        return action_index

    def get_state_dict(self, device: str = "cpu") -> Dict[str, Dict[str, torch.Tensor]]:
        """
        获取当前模型的状态字典
        参数:
            device: 目标设备（默认为"cpu"）
        返回:
            model_state_dict: 包含模型参数的状态字典
        """
        
        model_state_dict = {
            'actor': self.actor.module.state_dict() if isinstance(self.actor, nn.parallel.DistributedDataParallel) else self.actor.state_dict(),
            'total_it': self.total_it
        }
        
        # 将状态字典移动到指定设备
        for key in model_state_dict:
            if key != 'total_it':  # 跳过非张量值
                for param_key in model_state_dict[key]:
                    model_state_dict[key][param_key] = model_state_dict[key][param_key].to(device)
        
        return model_state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Dict[str, torch.Tensor]]):
        """
        加载模型的状态字典
        参数:
            state_dict: 包含模型参数的状态字典
        """
        
        # 加载状态字典到actor模型
        if isinstance(self.actor, nn.parallel.DistributedDataParallel):
            self.actor.module.load_state_dict(state_dict['actor'])
        else:
            self.actor.load_state_dict(state_dict['actor'])

        # 更新迭代次数
        self.total_it = state_dict.get('total_it', 0)


def initialize_bc(config, rank, world_size):
    """
    初始化DDP模式下的BC模型
    """
    device = torch.device(f"cuda:{rank}") if world_size > 1 else torch.device("cpu")

    # 创建网络（只需要actor）
    actor = Actor(config.node_dim, config.gnn_hidden_dim).to(device)

    # 包装成DDP模型
    if world_size > 1:
        actor = DDP(actor, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # 创建优化器（只需要actor的优化器）
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    
    # 创建BC实例
    model = DDPBC(
        actor=actor,
        actor_optimizer=actor_optimizer,
        device=device,
        rank=rank,
        world_size=world_size,
        clip_grad_norm=getattr(config, 'clip_grad_norm', 1.0),
    )
    
    if config.load_model_path is not None:
        # 加载模型参数
        state_dict = torch.load(config.load_model_path, map_location=device)
        model.load_state_dict(state_dict)
        
    return model


def create_bc_model(config, node_dim, rank=0, world_size=1):
    """创建BC模型"""
    return initialize_bc(config, node_dim, rank, world_size)
