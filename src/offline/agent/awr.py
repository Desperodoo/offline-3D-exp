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
from .model.sgformer import ValueNet
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
    mini_batch_size: int = 256  # AWR的mini-batch大小
    discount: float = 0.99  # 折扣因子
    clip_grad_norm: Optional[float] = 1.0  # 默认开启梯度裁剪
    actor_update_freq: int = 1  # Actor更新频率

    # AWR特有参数
    temperature: float = 3.0  # 温度参数，控制advantage加权的强度
    value_loss_type: str = "mse"  # Value损失类型: "mse" 或 "huber"
    advantage_type: str = "q_minus_v"  # Advantage计算方式: "q_minus_v", "gae", "td_error"
    weight_clipping: float = 20.0  # 权重裁剪上限
    tau: float = 0.005  # 目标网络更新率
    
    # Wandb日志
    project: str = "AWR-Graph"
    group: str = "AWR"
    name: str = "AWR"
    
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


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    计算Huber损失
    
    参数:
        pred: 预测值
        target: 目标值
        delta: Huber损失的阈值
        
    返回:
        Huber损失
    """
    abs_error = torch.abs(pred - target)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + delta * linear


class DDPAWR:
    """
    基于DDP的Advantage Weighted Regression (AWR)算法实现 - 图结构特化版本
    
    AWR核心思想：
    1. 训练Q网络和V网络（类似于Actor-Critic）
    2. 计算advantage = Q(s,a) - V(s)
    3. 使用advantage加权的行为克隆损失训练策略网络
    4. 权重 = exp(advantage / temperature)
    """
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_2: nn.Module,
        value_net: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2_optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        temperature: float = 3.0,
        value_loss_type: str = "mse",
        advantage_type: str = "q_minus_v",
        weight_clipping: float = 20.0,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        clip_grad_norm: float = 1.0,
        actor_update_freq: int = 1,
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer
        
        self.value_net = value_net
        self.value_optimizer = value_optimizer

        self.discount = discount
        self.tau = tau
        self.temperature = temperature
        self.value_loss_type = value_loss_type
        self.advantage_type = advantage_type
        self.weight_clipping = weight_clipping
        self.clip_grad_norm = clip_grad_norm
        self.actor_update_freq = actor_update_freq

        self.total_it = 0
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        # 保存初始学习率，用于学习率衰减
        self.initial_actor_lr = self._get_lr(self.actor_optimizer)
        self.initial_critic_lr = self._get_lr(self.critic_1_optimizer)
        self.initial_value_lr = self._get_lr(self.value_optimizer)

    def _get_lr(self, optimizer):
        """获取优化器的当前学习率"""
        for param_group in optimizer.param_groups:
            return param_group['lr']
        return None
    
    def get_current_lr(self):
        """获取当前学习率信息，用于日志记录"""
        return {
            "actor_lr": self._get_lr(self.actor_optimizer),
            "critic_lr": self._get_lr(self.critic_1_optimizer),
            "value_lr": self._get_lr(self.value_optimizer)
        }

    def train(self, batch: Dict[str, torch.Tensor], mini_batch: int = None) -> Dict[str, float]:
        """
        执行一步AWR训练更新
        
        参数:
            batch: 训练数据批次
            mini_batch: mini-batch大小，如果为None则使用整个batch
        """
        return self._train_awr(batch, mini_batch)

    def _compute_advantage(self, observations, actions, rewards, next_observations, dones):
        """
        计算advantage值
        
        参数:
            observations: 当前观察
            actions: 动作
            rewards: 奖励
            next_observations: 下一状态观察
            dones: 终止标志
            
        返回:
            advantage: advantage值
        """
        with torch.no_grad():
            if self.advantage_type == "q_minus_v":
                # Q(s,a) - V(s)
                q1 = self.critic_1(observations)
                q2 = self.critic_2(observations)
                q_val = torch.min(q1, q2)  # [batch_size, n_viewpoints, 1]
                
                actions_for_gather = actions.unsqueeze(1)  # [batch_size, 1]
                q_val = q_val.gather(1, actions_for_gather.unsqueeze(-1)).squeeze(-1)  # [batch_size, 1]
                
                v_val = self.value_net(observations).unsqueeze(1)  # [batch_size, 1]
                advantage = q_val - v_val
                
            elif self.advantage_type == "td_error":
                # TD error: r + γV(s') - V(s)
                v_val = self.value_net(observations).unsqueeze(1)
                next_v_val = self.value_net(next_observations).unsqueeze(1)
                
                target_v = rewards + (1 - dones) * self.discount * next_v_val
                advantage = target_v - v_val
                
            else:
                raise ValueError(f"不支持的advantage类型: {self.advantage_type}")
        
        return advantage

    def _compute_awr_weights(self, advantage):
        """
        计算AWR权重
        
        参数:
            advantage: advantage值 [batch_size, 1]
            
        返回:
            weights: AWR权重 [batch_size, 1]
        """
        # 使用指数函数计算权重: exp(advantage / temperature)
        weights = torch.exp(advantage / self.temperature)
        
        # 权重裁剪，避免过大的权重
        weights = torch.clamp(weights, max=self.weight_clipping)
        
        return weights

    def _train_awr(self, batch: Dict[str, torch.Tensor], mini_batch: int = None) -> Dict[str, float]:
        """
        执行一步AWR训练更新，包含梯度裁剪
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
        self.value_optimizer.zero_grad(set_to_none=True)
        self.critic_1_optimizer.zero_grad(set_to_none=True)
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        
        total_value_loss = 0.0
        total_q_loss = 0.0
        total_actor_loss = 0.0
        total_awr_weight = 0.0
        
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
            
            # ---------- 更新值网络 ----------
            # 计算目标V值
            with torch.no_grad():
                next_v = self.value_net(mb_next_observation).unsqueeze(1)
                target_v = mb_rewards + (1 - mb_dones) * self.discount * next_v
            
            v_pred = self.value_net(mb_current_observation).unsqueeze(1)
            
            if self.value_loss_type == "mse":
                value_loss = F.mse_loss(v_pred, target_v)
            elif self.value_loss_type == "huber":
                value_loss = huber_loss(v_pred, target_v).mean()
            else:
                raise ValueError(f"不支持的value损失类型: {self.value_loss_type}")
            
            value_loss.backward()
            total_value_loss += value_loss.item() * mb_size / batch_size
            
            # ---------- 更新Q网络 ----------
            # 使用双Q网络进行训练
            raw_q1 = self.critic_1(mb_current_observation)  # [batch_size, n_viewpoints, 1]
            raw_q2 = self.critic_2(mb_next_observation)  # [batch_size, n_viewpoints, 1]
            
            actions_for_gather = mb_actions.unsqueeze(1)  # [batch_size, 1]
            q1_pred = raw_q1.gather(1, actions_for_gather.unsqueeze(-1)).squeeze(-1)  # [batch_size, 1]
            q2_pred = raw_q2.gather(1, actions_for_gather.unsqueeze(-1)).squeeze(-1)  # [batch_size, 1]
            
            # 计算目标Q值
            with torch.no_grad():
                target_q1 = self.critic_1_target(mb_next_observation)  # [batch_size, n_viewpoints, 1]
                target_q2 = self.critic_2_target(mb_next_observation)  # [batch_size, n_viewpoints, 1]
                # 使用当前策略计算下一状态的动作分布
                next_action_logits = self.actor(mb_next_observation)
                next_action_probs = F.softmax(next_action_logits, dim=-1)  # [batch_size, n_viewpoints]
                
                # 压缩Q值维度以进行期望计算
                target_q1_2d = target_q1.squeeze(-1)  # [batch_size, n_viewpoints]
                target_q2_2d = target_q2.squeeze(-1)  # [batch_size, n_viewpoints]
                
                # 期望Q值
                next_q = torch.min(
                    (next_action_probs * target_q1_2d).sum(dim=1, keepdim=True),
                    (next_action_probs * target_q2_2d).sum(dim=1, keepdim=True)
                )
                target_q = mb_rewards + (1 - mb_dones) * self.discount * next_q
            
            q1_loss = F.mse_loss(q1_pred, target_q)
            q2_loss = F.mse_loss(q2_pred, target_q)
            q_loss = 0.5 * (q1_loss + q2_loss)
            q_loss.backward()
            total_q_loss += q_loss.item() * mb_size / batch_size
            
            # ---------- 更新策略网络（AWR） ----------
            if self.total_it % self.actor_update_freq == 0:
                # 计算advantage
                advantage = self._compute_advantage(
                    mb_current_observation, mb_actions, mb_rewards, 
                    mb_next_observation, mb_dones
                )
                
                # 计算AWR权重
                awr_weights = self._compute_awr_weights(advantage)
                
                # 策略输出
                action_logits = self.actor(mb_current_observation)
                
                # 计算加权的行为克隆损失
                bc_loss = F.cross_entropy(action_logits, mb_actions, reduction='none')
                bc_loss = bc_loss.unsqueeze(1)  # [batch_size, 1]
                
                # AWR损失：advantage加权的BC损失
                awr_loss = (awr_weights * bc_loss).mean()
                awr_loss.backward()
                
                total_actor_loss += awr_loss.item() * mb_size / batch_size
                total_awr_weight += awr_weights.mean().item() * mb_size / batch_size
                accumulate_actor += 1

        # 所有mini-batch处理完毕后，进行梯度裁剪并更新参数
        # 梯度裁剪
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.clip_grad_norm)
            if accumulate_actor > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        
        # 更新参数
        self.value_optimizer.step()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        
        if accumulate_actor > 0:
            self.actor_optimizer.step()
        
        # 记录训练信息
        log_dict["value_loss"] = total_value_loss
        log_dict["q_loss"] = total_q_loss
        log_dict["actor_loss"] = total_actor_loss
        log_dict["awr_weight"] = total_awr_weight
        
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
            
        # 更新value网络学习率
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = self.initial_value_lr * lr_factor
        
        # 返回当前学习率信息，用于记录
        return {
            "actor_lr": self.initial_actor_lr * lr_factor,
            "critic_lr": self.initial_critic_lr * lr_factor,
            "value_lr": self.initial_value_lr * lr_factor,
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
            'value_net': self.value_net.module.state_dict() if isinstance(self.value_net, nn.parallel.DistributedDataParallel) else self.value_net.state_dict(),
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

    def update_temperature(self, progress_remaining):
        """
        使用指数衰减策略更新温度参数
        
        参数:
            progress_remaining: 剩余训练进度比例 (0.0 到 1.0)
        
        返回:
            包含当前温度信息的字典
        """
        # 确保进度值合法
        progress_remaining = np.clip(progress_remaining, 0.0, 1.0)
        
        # 计算当前进度 (0到1之间)
        progress = 1.0 - progress_remaining
        
        # AWR模式：动态调整温度参数
        # 训练初期使用较高的温度（更平滑的权重），后期降低温度（更尖锐的权重）
        initial_temp = 3.0
        min_temp = 1.0
        decay_rate = 1.0
        
        # 指数衰减公式
        temp_factor = min_temp / initial_temp + (1 - min_temp / initial_temp) * np.exp(-decay_rate * progress)
        current_temp = initial_temp * temp_factor
        
        # 更新当前的温度参数
        self.temperature = current_temp
        
        # 返回温度信息，用于记录
        return {
            "temperature": current_temp,
            "temperature_factor": temp_factor,
            "temperature_progress": progress
        }

    def update_hyperparameters(self, progress_remaining):
        """
        统一的超参数更新函数，整合学习率和温度更新
        
        参数:
            progress_remaining: 剩余训练进度比例 (0.0 到 1.0)
        
        返回:
            包含所有超参数更新信息的字典
        """
        # 更新学习率
        lr_info = self.update_learning_rate(progress_remaining)
        
        # 更新温度
        temp_info = self.update_temperature(progress_remaining)
        
        # 合并所有更新信息
        update_info = {}
        
        # 添加学习率信息
        for k, v in lr_info.items():
            update_info[f"lr/{k}"] = v
        
        # 添加温度信息
        for k, v in temp_info.items():
            update_info[f"hyperparams/{k}"] = v
        
        return update_info


def initialize_awr(config, rank, world_size):
    """
    初始化DDP模式下的AWR/BC模型
    """
    device = torch.device(f"cuda:{rank}") if world_size > 1 else torch.device("cpu")

    # 创建网络
    actor = Actor(config.node_dim, config.gnn_hidden_dim).to(device)
    critic_1 = Critic(config.node_dim, config.gnn_hidden_dim).to(device)
    critic_2 = Critic(config.node_dim, config.gnn_hidden_dim).to(device)
    value_net = ValueNet(config.node_dim, config.gnn_hidden_dim).to(device)

    # 包装成DDP模型
    if world_size > 1:
        actor = DDP(actor, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        critic_1 = DDP(critic_1, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        critic_2 = DDP(critic_2, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        value_net = DDP(value_net, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # 创建优化器
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=config.learning_rate)
    
    # 创建AWR实例
    model = DDPAWR(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        value_net=value_net,
        value_optimizer=value_optimizer,
        temperature=config.temperature,
        value_loss_type=config.value_loss_type,
        advantage_type=config.advantage_type,
        weight_clipping=config.weight_clipping,
        device=device,
        discount=getattr(config, 'discount', 0.99),
        tau=getattr(config, 'tau', 0.005),
        rank=rank,
        world_size=world_size,
        clip_grad_norm=getattr(config, 'clip_grad_norm', 1.0),
        actor_update_freq=getattr(config, 'actor_update_freq', 1),
    )
    
    if config.load_model_path is not None:
        # 加载模型参数
        state_dict = torch.load(config.load_model_path, map_location=device)
        model.load_state_dict(state_dict)
        
    return model
