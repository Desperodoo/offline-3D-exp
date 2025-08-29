import os
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from typing import Dict, List, Optional
from dataclasses import asdict, dataclass
from .model.sgformer import PolicyNet as Actor
from .model.sgformer import QNet as Critic
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
    mini_batch_size: int = 256  # CQL的mini-batch大小
    discount: float = 0.99  # 折扣因子
    clip_grad_norm: Optional[float] = 1.0  # 默认开启梯度裁剪
    actor_update_freq: int = 2  # Actor更新频率

    # CQL特有参数
    cql_alpha: float = 5.0  # CQL损失权重
    cql_min_q_weight: float = 10.0  # CQL最小Q值权重
    cql_temp: float = 1.0  # CQL温度参数
    cql_max_target_backup: bool = True  # 是否使用最大目标备份
    cql_clip_diff_min: float = -np.inf  # CQL差值裁剪最小值
    cql_clip_diff_max: float = np.inf  # CQL差值裁剪最大值
    tau: float = 0.005  # 目标网络更新率
    
    # Wandb日志
    project: str = "CQL-Graph"
    group: str = "CQL"
    name: str = "CQL"
    
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
            

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """软更新目标网络参数"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class DDPCQL:
    """
    基于DDP的保守Q学习(CQL)算法实现 - 图结构特化版本
    """
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_2: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        cql_alpha: float = 5.0,
        cql_min_q_weight: float = 10.0,
        cql_temp: float = 1.0,
        cql_max_target_backup: bool = True,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        clip_grad_norm: float = 1.0,
        actor_update_freq: int = 2,
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.discount = discount
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.cql_min_q_weight = cql_min_q_weight
        self.cql_temp = cql_temp
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.clip_grad_norm = clip_grad_norm
        self.actor_update_freq = actor_update_freq

        self.total_it = 0
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        # 保存初始学习率，用于学习率衰减
        self.initial_actor_lr = self._get_lr(self.actor_optimizer)
        self.initial_critic_lr = self._get_lr(self.critic_1_optimizer)

    def _get_lr(self, optimizer):
        """获取优化器的当前学习率"""
        for param_group in optimizer.param_groups:
            return param_group['lr']
        return None
    
    def get_current_lr(self):
        """获取当前学习率信息，用于日志记录"""
        return {
            "actor_lr": self._get_lr(self.actor_optimizer),
            "critic_lr": self._get_lr(self.critic_1_optimizer)
        }

    def train(self, batch: Dict[str, torch.Tensor], mini_batch: int = None) -> Dict[str, float]:
        """
        执行一步CQL训练更新
        
        参数:
            batch: 训练数据批次
            mini_batch: mini-batch大小，如果为None则使用整个batch
        """
        return self._train_cql(batch, mini_batch)

    def _compute_cql_loss(self, q_values, current_actions, observations):
        """
        计算CQL损失
        
        参数:
            q_values: Q值 [batch_size, n_viewpoints, 1]
            current_actions: 当前动作 [batch_size]
            observations: 观察值
        
        返回:
            cql_loss: CQL损失
        """
        # 处理Q值的维度：[batch_size, n_viewpoints, 1] -> [batch_size, n_viewpoints]
        q_values_2d = q_values.squeeze(-1)  # [batch_size, n_viewpoints]
        
        batch_size = q_values_2d.shape[0]
        action_dim = q_values_2d.shape[1]
        
        # 计算所有动作的Q值的logsumexp
        logsumexp_q = torch.logsumexp(q_values_2d / self.cql_temp, dim=1, keepdim=True)
        
        # 计算数据中动作的Q值
        current_q = q_values_2d.gather(1, current_actions.unsqueeze(1))
        
        # CQL损失：鼓励数据外动作的Q值小于数据内动作的Q值
        cql_loss = (logsumexp_q * self.cql_temp - current_q).mean()
        
        return cql_loss

    def _train_cql(self, batch: Dict[str, torch.Tensor], mini_batch: int = None) -> Dict[str, float]:
        """
        执行一步CQL训练更新，包含梯度裁剪
        支持mini-batch训练：将一个大batch分成多个小batch进行前向和反向传播，最后统一进行梯度裁剪和参数更新
        
        参数:
            batch: 训练数据批次
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
        
        # 清空所有梯度
        self.critic_1_optimizer.zero_grad(set_to_none=True)
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        
        total_q_loss = 0.0
        total_cql_loss = 0.0
        total_actor_loss = 0.0
        
        accumulate_actor = 0  # 跟踪actor累计更新次数
        
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
            
            mb_next_observation = [
                batch['next_node_inputs'][beg:end], 
                batch['next_node_padding_mask'][beg:end], 
                batch['next_current_index'][beg:end], 
                batch['next_viewpoints'][beg:end], 
                batch['next_viewpoint_padding_mask'][beg:end], 
                batch['next_adj_list'][beg:end]
            ]
            
            mb_actions = batch['actions'][beg:end]
            mb_rewards = batch['rewards'][beg:end].unsqueeze(1)
            mb_dones = batch['dones'][beg:end].unsqueeze(1).float()
            
            # ---------- 更新Q网络 ----------
            # 计算当前Q值
            raw_q1 = self.critic_1(mb_current_observation)  # [batch_size, n_viewpoints, 1]
            raw_q2 = self.critic_2(mb_current_observation)  # [batch_size, n_viewpoints, 1]
            
            # 正确处理Q网络输出的维度：[batch_size, n_viewpoints, 1]
            actions_for_gather = mb_actions.unsqueeze(1)  # [batch_size, 1]
            q1_pred = raw_q1.gather(1, actions_for_gather.unsqueeze(-1)).squeeze(-1)  # [batch_size, 1]
            q2_pred = raw_q2.gather(1, actions_for_gather.unsqueeze(-1)).squeeze(-1)  # [batch_size, 1]
            
            # 计算目标Q值
            with torch.no_grad():
                if self.cql_max_target_backup:
                    # 使用最大目标备份
                    next_q1 = self.critic_1_target(mb_next_observation)  # [batch_size, n_viewpoints, 1]
                    next_q2 = self.critic_2_target(mb_next_observation)  # [batch_size, n_viewpoints, 1]
                    next_q = torch.max(next_q1, next_q2).max(dim=1, keepdim=True)[0]  # [batch_size, 1, 1]
                    next_q = next_q.squeeze(-1)  # [batch_size, 1]
                else:
                    # 简化版本：使用双Q网络的最小值和Actor选择最优动作
                    next_q1 = self.critic_1_target(mb_next_observation)  # [batch_size, n_viewpoints, 1]
                    next_q2 = self.critic_2_target(mb_next_observation)  # [batch_size, n_viewpoints, 1]
                    next_q = torch.min(next_q1, next_q2)  # [batch_size, n_viewpoints, 1]
                    
                    # 使用Actor网络选择最优动作
                    next_action_logits = self.actor(mb_next_observation)  # [batch_size, num_viewpoints]
                    next_actions = torch.argmax(next_action_logits, dim=1)  # [batch_size]
                    
                    # 确保动作索引不超出Q网络输出维度
                    max_q_dim = next_q.shape[1]  # n_viewpoints
                    next_actions = torch.clamp(next_actions, 0, max_q_dim - 1)
                    
                    # 选择对应动作的Q值，正确处理维度
                    next_actions_for_gather = next_actions.unsqueeze(1)  # [batch_size, 1]
                    next_q = next_q.gather(1, next_actions_for_gather.unsqueeze(-1)).squeeze(-1)  # [batch_size, 1]
                
                target_q = mb_rewards + (1 - mb_dones) * self.discount * next_q
                target_q = torch.clamp(target_q, self.cql_clip_diff_min, self.cql_clip_diff_max)
            
            # 标准贝尔曼损失
            q1_loss = F.mse_loss(q1_pred, target_q)
            q2_loss = F.mse_loss(q2_pred, target_q)
            q_loss = 0.5 * (q1_loss + q2_loss)
            
            # CQL损失
            cql_loss_1 = self._compute_cql_loss(raw_q1, mb_actions, mb_current_observation)
            cql_loss_2 = self._compute_cql_loss(raw_q2, mb_actions, mb_current_observation)
            cql_loss = 0.5 * (cql_loss_1 + cql_loss_2)
            
            # 总的critic损失
            total_critic_loss = q_loss + self.cql_alpha * cql_loss
            total_critic_loss.backward()
            
            total_q_loss += q_loss.item() * mb_size / batch_size
            total_cql_loss += cql_loss.item() * mb_size / batch_size
            
            # ---------- 更新策略网络（根据频率） ----------
            if self.total_it % self.actor_update_freq == 0:
                # 计算策略损失（最大化Q值）
                action_logits = self.actor(mb_current_observation)  # [batch_size, n_viewpoints]
                action_probs = F.softmax(action_logits, dim=-1)  # [batch_size, n_viewpoints]
                
                # 使用当前的Q网络（不使用梯度停止）
                current_q1 = self.critic_1(mb_current_observation)  # [batch_size, n_viewpoints, 1]
                current_q2 = self.critic_2(mb_current_observation)  # [batch_size, n_viewpoints, 1]
                current_q = torch.min(current_q1, current_q2)  # [batch_size, n_viewpoints, 1]
                
                # 处理Q值维度以匹配action_probs
                current_q = current_q.squeeze(-1)  # [batch_size, n_viewpoints]
                
                # 策略损失：最大化期望Q值
                actor_loss = -(action_probs * current_q).sum(dim=1).mean()
                actor_loss.backward()
                
                total_actor_loss += actor_loss.item() * mb_size / batch_size
                accumulate_actor += 1

        # 所有mini-batch处理完毕后，进行梯度裁剪并更新参数
        # 梯度裁剪
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.clip_grad_norm)
            if accumulate_actor > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        
        # 更新参数
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        
        if accumulate_actor > 0:
            self.actor_optimizer.step()
        
        # 记录训练信息
        log_dict["q_loss"] = total_q_loss
        log_dict["cql_loss"] = total_cql_loss
        log_dict["actor_loss"] = total_actor_loss
        
        # 软更新目标网络
        soft_update(self.critic_1_target, self.critic_1, self.tau)
        soft_update(self.critic_2_target, self.critic_2, self.tau)
        
        return log_dict

    def update_learning_rate(self, progress_remaining):
        """
        使用玻尔兹曼退火(Boltzmann annealing)策略更新学习率
        
        参数:
            progress_remaining: 剩余训练进度比例 (0.0 到 1.0)
        """
        # 确保进度值合法
        progress_remaining = np.clip(progress_remaining, 0.00001, 1.0)
        
        # 计算当前进度 (0到1之间)
        progress = 1.0 - progress_remaining
        
        # 玻尔兹曼退火参数
        initial_temperature = 1.0  # 初始温度
        min_lr_factor = 0.01      # 最小学习率因子
        
        # 计算当前温度: T(t) = T0 / log(e + t)
        # 使用e而不是1作为基础，确保初始值不会过大
        normalized_progress = progress * 20  # 将进度扩展到更大范围，使温度下降更明显
        current_temperature = initial_temperature / np.log(np.e + normalized_progress) 
        
        # 计算学习率因子 (范围在min_lr_factor到1.0之间)
        # 随着训练进行，温度降低，学习率因子也降低
        lr_factor = min_lr_factor + (1.0 - min_lr_factor) * (current_temperature / initial_temperature)
        
        # 更新actor学习率
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.initial_actor_lr * lr_factor
            
        # 更新critic学习率
        for param_group in self.critic_1_optimizer.param_groups:
            param_group['lr'] = self.initial_critic_lr * lr_factor
        for param_group in self.critic_2_optimizer.param_groups:
            param_group['lr'] = self.initial_critic_lr * lr_factor
        
        # 返回当前学习率信息，用于记录
        return {
            "actor_lr": self.initial_actor_lr * lr_factor,
            "critic_lr": self.initial_critic_lr * lr_factor,
            "lr_factor": lr_factor,
            "temperature": current_temperature
        }

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
            'critic_1': self.critic_1.module.state_dict() if isinstance(self.critic_1, nn.parallel.DistributedDataParallel) else self.critic_1.state_dict(),
            'critic_2': self.critic_2.module.state_dict() if isinstance(self.critic_2, nn.parallel.DistributedDataParallel) else self.critic_2.state_dict(),
            'critic_1_target': self.critic_1_target.state_dict(),
            'critic_2_target': self.critic_2_target.state_dict(),
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
        
        # 加载状态字典到各个模型
        if isinstance(self.actor, nn.parallel.DistributedDataParallel):
            self.actor.module.load_state_dict(state_dict['policy'])
        else:
            self.actor.load_state_dict(state_dict['policy'])

    def update_cql_alpha(self, progress_remaining):
        """
        使用指数衰减策略更新cql_alpha
        
        参数:
            progress_remaining: 剩余训练进度比例 (0.0 到 1.0)
        
        返回:
            包含当前cql_alpha信息的字典
        """
        # 确保进度值合法
        progress_remaining = np.clip(progress_remaining, 0.0, 1.0)
        
        # 计算当前进度 (0到1之间)
        progress = 1.0 - progress_remaining
        
        # CQL模式：在训练初期使用较高的CQL权重，后期逐渐减少
        # 这样可以在初期更强地约束Q值，后期更多地关注策略性能
        initial_alpha = 5.0
        min_alpha = 1.0
        decay_rate = 1.5
        
        # 指数衰减公式
        alpha_factor = min_alpha / initial_alpha + (1 - min_alpha / initial_alpha) * np.exp(-decay_rate * progress)
        current_alpha = initial_alpha * alpha_factor
        
        # 更新当前的cql_alpha
        self.cql_alpha = current_alpha
        
        # 返回衰减信息，用于记录
        return {
            "cql_alpha": current_alpha,
            "cql_alpha_factor": alpha_factor,
            "cql_alpha_progress": progress
        }

    def update_hyperparameters(self, progress_remaining):
        """
        统一的超参数更新函数，整合学习率和CQL alpha更新
        
        参数:
            progress_remaining: 剩余训练进度比例 (0.0 到 1.0)
        
        返回:
            包含所有超参数更新信息的字典
        """
        # 更新学习率
        lr_info = self.update_learning_rate(progress_remaining)
        
        # 更新cql_alpha
        alpha_info = self.update_cql_alpha(progress_remaining)
        
        # 合并所有更新信息
        update_info = {}
        
        # 添加学习率信息
        for k, v in lr_info.items():
            update_info[f"lr/{k}"] = v
        
        # 添加CQL alpha信息
        for k, v in alpha_info.items():
            update_info[f"hyperparams/{k}"] = v
        
        return update_info


def initialize_cql(config, rank, world_size):
    """
    初始化DDP模式下的CQL/BC模型
    """
    device = torch.device(f"cuda:{rank}") if world_size > 1 else torch.device("cpu")

    # 创建网络
    actor = Actor(config.node_dim, config.gnn_hidden_dim).to(device)
    critic_1 = Critic(config.node_dim, config.gnn_hidden_dim).to(device)
    critic_2 = Critic(config.node_dim, config.gnn_hidden_dim).to(device)

    # 包装成DDP模型
    if world_size > 1:
        actor = DDP(actor, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        critic_1 = DDP(critic_1, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        critic_2 = DDP(critic_2, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # 创建优化器
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)
    
    # 创建CQL实例
    model = DDPCQL(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        cql_alpha=config.cql_alpha,
        cql_min_q_weight=config.cql_min_q_weight,
        cql_temp=config.cql_temp,
        cql_max_target_backup=config.cql_max_target_backup,
        cql_clip_diff_min=config.cql_clip_diff_min,
        cql_clip_diff_max=config.cql_clip_diff_max,
        device=device,
        discount=getattr(config, 'discount', 0.99),
        tau=getattr(config, 'tau', 0.005),
        rank=rank,
        world_size=world_size,
        clip_grad_norm=getattr(config, 'clip_grad_norm', 1.0),
        actor_update_freq=getattr(config, 'actor_update_freq', 2),
    )
    
    if config.load_model_path is not None:
        # 加载模型参数
        state_dict = torch.load(config.load_model_path, map_location=device)
        model.load_state_dict(state_dict)
        
    return model
