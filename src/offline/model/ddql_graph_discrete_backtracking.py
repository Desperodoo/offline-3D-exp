import copy, torch, math, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP

from model.graph_denoiser import GraphActionDenoiser, create_pointer_denoiser, create_simple_denoiser
from model.diffusion_discrete import DiscreteDiffusion
from model.sgformer import QNet                           # graph critic
from dataclasses import dataclass
from typing import Dict, Optional
import uuid
import os


@dataclass
class TrainConfig:
    """Diffusion-QL训练配置类"""
    
    # 实验设置
    device: str = "cuda"
    seed: int = 0  
    eval_freq: int = int(1e2)  # 评估频率（步数）
    eval_start: int = 0  # 开始评估的步数，默认从第0步开始
    max_timesteps: int = int(3e5)  # 运行环境的最大时间步数
    checkpoints_path: Optional[str] = None  # 保存路径
    load_model_path: Optional[str] = None  # 模型加载路径
    
    # 缓冲区设置
    buffer_size: int = 1_000_000  # 回放缓冲区大小
    
    # 优化设置
    learning_rate: float = 1e-4  # DDQL使用统一学习率
    batch_size: int = 256  # 训练批量大小
    mini_batch_size: int = None  # mini-batch大小，None表示使用整个batch
    discount: float = 0.99  # 折扣因子
    clip_grad_norm: float = 1.0  # 梯度裁剪
    
    # DDQL特有参数
    actor_bc_coef: float = 1.0  # 行为克隆损失权重（原eta参数）
    n_timesteps: int = 20  # 扩散步数
    ema_decay: float = 0.995  # EMA衰减率
    step_start_ema: int = 1000  # 开始EMA的步数
    update_ema_every: int = 5  # EMA更新频率
    tau: float = 0.005  # 目标网络软更新系数
    actor_update_freq: int = 1  # Actor更新频率
    
    # 新增：扩散模式选择
    diffusion_mode: str = "pointer"  # "pointer" 或 "simple"
    use_fixed_actions: bool = False  # 是否使用固定动作空间（简化模式）
    
    # Wandb日志
    project: str = "DDQL-Graph"
    group: str = "DDQL"
    name: str = "DDQL"
    
    # 图相关参数
    node_dim: int = 6  # 节点特征维度
    num_nodes: int = None  # 图中节点数量
    edge_dim: int = None  # 边特征维度
    gnn_hidden_dim: int = 128  # 图神经网络隐藏层维度
    viewpoint_padding_size: int = 180  # 视点填充大小/动作空间维度
    
    def __post_init__(self):
        """初始化后处理"""
        # 根据diffusion_mode设置use_fixed_actions
        if self.diffusion_mode == "simple":
            self.use_fixed_actions = True
        elif self.diffusion_mode == "pointer":
            self.use_fixed_actions = False
        else:
            raise ValueError(f"不支持的diffusion模式: {self.diffusion_mode}")
            
        self.name = f"{self.name}-{self.diffusion_mode}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
    
    def update(self, **kwargs):
        """更新配置参数"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"警告: 配置中不存在参数 '{k}'")
        return self

    def display(self):
        """显示当前配置"""
        print("====== DDQL-Graph训练配置 ======")
        for k, v in sorted(vars(self).items()):
            print(f"{k}: {v}")
        print("================================")


# ---------------------------- 双 Q 网络封装 ---------------------------- #
class TwinQ(nn.Module):
    """Q1、Q2 两个独立的 Graph QNet，输出 [B,K]"""
    def __init__(self, node_dim, embed_dim):
        super().__init__()
        self.q1 = QNet(node_dim, embed_dim)
        self.q2 = QNet(node_dim, embed_dim)

    def forward(self, obs):
        return self.q1(obs).squeeze(-1), self.q2(obs).squeeze(-1)  # each => [B,K]

    def q_min(self, obs, act_idx):
        q1, q2 = self.forward(obs)
        q1 = q1.gather(1, act_idx.unsqueeze(1))
        q2 = q2.gather(1, act_idx.unsqueeze(1))
        return torch.min(q1, q2)                    # [B,1]


# ---------------------------- EMA 辅助 ---------------------------- #
class EMA:
    def __init__(self, beta: float):
        self.beta = beta

    @torch.no_grad()
    def update(self, ema_m, cur_m):
        for p_ema, p_cur in zip(ema_m.parameters(), cur_m.parameters()):
            p_ema.data.mul_(self.beta).add_(p_cur.data, alpha=1 - self.beta)


# ======================= 主算法：DiffusionGraphQL ===================== #
class DiffusionGraphQL:
    def __init__(
        self,
        actor: nn.Module,
        actor_opt: torch.optim.Optimizer,
        critic: nn.Module,
        critic_opt: torch.optim.Optimizer,
        device: str,
        discount: float     = 0.99,
        tau: float          = 0.005,
        actor_bc_coef: float = 1.0,        # 行为克隆损失权重（原eta参数）
        ema_decay: float    = 0.995,
        step_start_ema: int = 1000,
        update_ema_every: int = 5,
        grad_clip: float    = 1.0,
        rank: int           = 0,          # 新增: 当前进程的rank
        world_size: int     = 1,          # 新增: 总进程数
        actor_update_freq: int = 1,       # 新增: Actor更新频率
    ):
        self.device = device
        self.rank = rank
        self.world_size = world_size

        # ---------- Actor : Discrete Diffusion ---------- #
        self.actor = actor
        self.actor_opt = actor_opt

        # ---------- Critic : Twin Graph QNet ---------- #
        self.critic = critic
        self.critic_tgt = copy.deepcopy(self.critic)
        self.critic_opt = critic_opt

        # ---------- Actor & Critic Module ---------- #
        # 添加这些行以保存原始模型引用
        self.actor_module = actor.module if isinstance(actor, nn.parallel.DistributedDataParallel) else actor
        self.critic_module = critic.module if isinstance(critic, nn.parallel.DistributedDataParallel) else critic
        self.critic_tgt_module = self.critic_tgt.module if isinstance(self.critic_tgt, nn.parallel.DistributedDataParallel) else self.critic_tgt

        # ---------- EMA ---------- #
        self.ema       = EMA(ema_decay)
        self.total_it      = 0
        self.start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.actor_update_freq = actor_update_freq

        # ---------- Hyper-params ---------- #
        self.discount = discount
        self.tau      = tau
        self.actor_bc_coef = actor_bc_coef  # 重命名为actor_bc_coef
        self.grad_clip = grad_clip
        
        # 保存初始学习率，用于学习率衰减
        self.initial_actor_lr = self._get_lr(self.actor_opt)
        self.initial_critic_lr = self._get_lr(self.critic_opt)
        
        # 保存初始actor_bc_coef用于衰减
        self.initial_actor_bc_coef = actor_bc_coef

    # ---------------------- 训练一步 ---------------------- #
    def train(self, batch, mini_batch: int = None):
        """
        执行一步训练更新，接受批次数据字典而非replay_buffer
        
        参数:
            batch: 训练数据批次字典
            mini_batch: mini-batch大小，如果为None则使用整个batch
        """
        self.total_it += 1
        log = {}
        
        # 提取批次数据
        obs = [
            batch['node_inputs'],
            batch['node_padding_mask'],
            batch['current_index'],
            batch['viewpoints'],
            batch['viewpoint_padding_mask'],
            batch['adj_list']
        ]
        
        next_obs = [
            batch['next_node_inputs'],
            batch['next_node_padding_mask'],
            batch['next_current_index'],
            batch['next_viewpoints'],
            batch['next_viewpoint_padding_mask'],
            batch['next_adj_list']
        ]
        
        act_idx = batch['actions']
        reward = batch['rewards'].unsqueeze(1)
        not_done = (~batch['dones']).float().unsqueeze(1)
        
        batch_size = act_idx.shape[0]
        
        # 确定是否使用mini-batch以及如何划分
        if mini_batch is None or mini_batch >= batch_size:
            slices = [(0, batch_size)]
        else:
            slices = [(i, min(i + mini_batch, batch_size)) for i in range(0, batch_size, mini_batch)]
            
        # 清空所有梯度
        self.critic_opt.zero_grad(set_to_none=True)
        if self.total_it % self.actor_update_freq == 0:
            self.actor_opt.zero_grad(set_to_none=True)
            
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_bc_loss = 0.0
        total_ql_loss = 0.0
        
        # 对每个mini-batch进行处理
        for beg, end in slices:
            mb_size = end - beg
            
            # 提取mini-batch数据
            mb_obs = [o[beg:end] for o in obs]
            mb_next_obs = [o[beg:end] for o in next_obs]
            mb_act_idx = act_idx[beg:end]
            mb_reward = reward[beg:end]
            mb_not_done = not_done[beg:end]
            
            # ========== Critic update ========= #
            q1, q2 = self.critic(mb_obs)          # each [B,K]，现在是二维
            q1a = q1.gather(1, mb_act_idx.unsqueeze(1))  # [B,1]，不需要再squeeze
            q2a = q2.gather(1, mb_act_idx.unsqueeze(1))

            with torch.no_grad():
                next_act = self.actor_module.sample(mb_next_obs, mb_next_obs[4], stochastic=False)         # [B]
                q1_t, q2_t = self.critic_tgt(mb_next_obs)
                q1_next = q1_t.gather(1, next_act.unsqueeze(1))   # 不需要再squeeze
                q2_next = q2_t.gather(1, next_act.unsqueeze(1))
                target_q = mb_reward + mb_not_done * self.discount * torch.min(q1_next, q2_next)

            critic_loss = F.mse_loss(q1a, target_q) + F.mse_loss(q2a, target_q)
            critic_loss.backward()
            
            total_critic_loss += critic_loss.item() * mb_size / batch_size

            # ========== Actor update (根据频率) ========= #
            if self.total_it % self.actor_update_freq == 0:
                bc_loss, diff_loss_dict = self.actor_module.loss(mb_act_idx, mb_obs[4], mb_obs)
                new_act = self.actor_module.sample(mb_obs, mb_obs[4], stochastic=False)                   # [B]
                q1_new = self.critic_module.q_min(mb_obs, new_act)
                
                # 归一化 trick（取另一条 Q 除绝对值）
                with torch.no_grad():
                    q_other = self.critic_module.q_min(mb_obs, mb_act_idx)      # baseline
                ql_loss = - q1_new.mean() / (q_other.abs().mean() + 1e-6)
                
                # 统一语义：actor_bc_coef控制BC损失权重，QL损失权重固定为1
                actor_loss = ql_loss + self.actor_bc_coef * bc_loss

                actor_loss.backward()
                
                total_actor_loss += actor_loss.item() * mb_size / batch_size
                total_bc_loss += bc_loss.item() * mb_size / batch_size
                total_ql_loss += ql_loss.item() * mb_size / batch_size
        
        # 所有mini-batch处理完后，进行梯度裁剪和参数更新
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            if self.total_it % self.actor_update_freq == 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                
        # 更新参数
        self.critic_opt.step()
        if self.total_it % self.actor_update_freq == 0:
            self.actor_opt.step()
            log["actor_loss"] = total_actor_loss
            log["bc_loss"] = total_bc_loss
            log["q_loss"] = total_ql_loss

        log["value_loss"] = total_critic_loss
        log.update(diff_loss_dict)
        
        # ---------- EMA & target-net ---------- #
        if self.total_it >= self.start_ema and self.total_it % self.update_ema_every == 0:
            self.ema.update(self.actor, self.actor)  # keeps same weights for clarity
        
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                p_t.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

        return log

    def _get_lr(self, optimizer):
        """获取优化器的当前学习率"""
        for param_group in optimizer.param_groups:
            return param_group['lr']
        return None

    def update_learning_rate(self, progress_remaining):
        """
        使用玻尔兹曼退火策略更新学习率
        
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
        normalized_progress = progress * 20
        current_temperature = initial_temperature / np.log(np.e + normalized_progress) 
        
        # 计算学习率因子 (范围在min_lr_factor到1.0之间)
        lr_factor = min_lr_factor + (1.0 - min_lr_factor) * (current_temperature / initial_temperature)
        
        # 更新actor学习率
        for param_group in self.actor_opt.param_groups:
            param_group['lr'] = self.initial_actor_lr * lr_factor
            
        # 更新critic学习率
        for param_group in self.critic_opt.param_groups:
            param_group['lr'] = self.initial_critic_lr * lr_factor
        
        # 返回当前学习率信息，用于记录
        return {
            "actor_lr": self.initial_actor_lr * lr_factor,
            "critic_lr": self.initial_critic_lr * lr_factor,
            "lr_factor": lr_factor,
            "temperature": current_temperature
        }
    
    def update_actor_bc_coef(self, progress_remaining):
        """
        使用指数衰减策略更新actor_bc_coef (与IQL保持一致)
        
        参数:
            progress_remaining: 剩余训练进度比例 (0.0 到 1.0)
        
        返回:
            包含当前actor_bc_coef信息的字典
        """
        # 确保进度值合法
        progress_remaining = np.clip(progress_remaining, 0.0, 1.0)
        
        # 计算当前进度 (0到1之间)
        progress = 1.0 - progress_remaining
        
        # DDQL模式：使用指数衰减，在训练后期减少BC损失权重，让QL损失起主导作用
        # 与IQL保持一致的衰减策略
        decay_rate = 2.0  # 衰减速率，值越大衰减越快
        min_coef_factor = 0.1  # 最小系数因子，防止BC损失完全消失
        
        # 指数衰减公式: coef = initial_coef * (min_factor + (1 - min_factor) * exp(-decay_rate * progress))
        decay_factor = min_coef_factor + (1 - min_coef_factor) * np.exp(-decay_rate * progress)
        current_coef = self.initial_actor_bc_coef * decay_factor
        
        # 更新当前的actor_bc_coef
        self.actor_bc_coef = current_coef
        
        # 返回衰减信息，用于记录
        return {
            "actor_bc_coef": current_coef,
            "actor_bc_coef_decay_factor": decay_factor,
            "actor_bc_coef_progress": progress
        }

    def get_current_lr(self):
        """获取当前学习率信息，用于日志记录"""
        return {
            "actor_lr": self._get_lr(self.actor_opt),
            "critic_lr": self._get_lr(self.critic_opt)
        }

    # ------------------- 推理接口 ------------------- #
    @torch.no_grad()
    def select_action(self, observation) -> torch.Tensor:
        """
        观察值 -> 动作序号
        改为返回Tensor以与IQL接口对齐
        """
        # 确保输入格式统一
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
            elif obs.dim() == 1 and i in [1, 4]:  # padding_masks
                processed_observation[i] = obs.unsqueeze(0).unsqueeze(0)
            elif obs.dim() == 0 and i == 2:  # current_index
                processed_observation[i] = obs.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif obs.dim() == 1 and i == 3:  # viewpoints
                processed_observation[i] = obs.unsqueeze(0).unsqueeze(-1)
            elif obs.dim() == 2 and i == 5:  # adj_list
                processed_observation[i] = obs.unsqueeze(0)
                
        act = self.actor_module.sample(processed_observation, processed_observation[4], stochastic=False)  # [1]
        return act

    # ---------------- Load / State Dict --------------- #

    def load_state_dict(self, state_dict: Dict[str, Dict[str, torch.Tensor]]):
        """
        加载模型的状态字典
        参数:
            state_dict: 包含模型参数的状态字典
        """
        if isinstance(self.actor, nn.parallel.DistributedDataParallel):
            self.actor.module.load_state_dict(state_dict["actor"])
        else:
            self.actor.load_state_dict(state_dict["actor"])
        if isinstance(self.critic, nn.parallel.DistributedDataParallel):
            self.critic.module.load_state_dict(state_dict["critic"])
        else:
            self.critic.load_state_dict(state_dict["critic"])
        if isinstance(self.critic_tgt, nn.parallel.DistributedDataParallel):
            self.critic_tgt.module.load_state_dict(state_dict["critic_tgt"])
        else:
            self.critic_tgt.load_state_dict(state_dict["critic_tgt"])
        # 更新总步数
        self.total_it = state_dict["total_it"]

    def get_state_dict(self, device: str = "cpu"):
        """
        获取当前模型的状态字典，用于保存和DDP训练
        
        参数:
            device: 目标设备（默认为"cpu"）
        返回:
            包含模型参数的状态字典
        """
        # 处理可能的DDP包装
        actor_state = self.actor.module.state_dict() if isinstance(self.actor, nn.parallel.DistributedDataParallel) else self.actor.state_dict()
        critic_state = self.critic.module.state_dict() if isinstance(self.critic, nn.parallel.DistributedDataParallel) else self.critic.state_dict()
        critic_tgt_state = self.critic_tgt.module.state_dict() if isinstance(self.critic_tgt, nn.parallel.DistributedDataParallel) else self.critic_tgt.state_dict()
        
        model_state_dict = {
            'actor': actor_state,
            'critic': critic_state,
            'critic_tgt': critic_tgt_state,
            'total_it': self.total_it
        }
        
        # 将状态字典移动到指定设备
        for key in model_state_dict:
            if key != 'total_it':  # 跳过非张量值
                for param_key in model_state_dict[key]:
                    model_state_dict[key][param_key] = model_state_dict[key][param_key].to(device)
        
        return model_state_dict


# 添加初始化函数，与DDPIQL的initialize_iql接口保持一致
def initialize_ddql(config, node_dim, rank, world_size):
    """
    初始化DDP模式下的DDQL模型，支持两种diffusion模式
    """
    device = torch.device(config.device) if hasattr(config, 'device') else (torch.device(f"cuda:{rank}") if world_size > 1 else torch.device("cpu"))
    
    # 根据配置选择denoiser类型
    if config.diffusion_mode == "pointer":
        denoiser = create_pointer_denoiser(
            node_dim=node_dim,
            embed_dim=config.gnn_hidden_dim,
            max_actions=config.viewpoint_padding_size,
            T=config.n_timesteps
        ).to(device)
        print(f"使用Pointer Network扩散模式")
    elif config.diffusion_mode == "simple":
        denoiser = create_simple_denoiser(
            node_dim=node_dim,
            embed_dim=config.gnn_hidden_dim,
            max_actions=config.viewpoint_padding_size,
            T=config.n_timesteps
        ).to(device)
        print(f"使用Simple扩散模式")
    else:
        raise ValueError(f"不支持的diffusion模式: {config.diffusion_mode}")

    # 创建Actor(Diffusion)，传入模式配置
    actor = DiscreteDiffusion(
        num_actions=config.viewpoint_padding_size,
        model=denoiser,
        T=config.n_timesteps,
        use_fixed_actions=config.use_fixed_actions
    ).to(device)
    
    # 创建Critic
    critic = TwinQ(node_dim, config.gnn_hidden_dim).to(device)
    
    # 包装成DDP模型
    if world_size > 1:
        actor = DDP(actor, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        critic = DDP(critic, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        
    # 创建优化器
    actor_opt = Adam(actor.parameters(), lr=config.learning_rate)
    critic_opt = Adam(critic.parameters(), lr=config.learning_rate)
        
    # 创建DiffusionQL模型
    model = DiffusionGraphQL(
        actor=actor,
        actor_opt=actor_opt,
        critic=critic,
        critic_opt=critic_opt,
        device=device,
        discount=getattr(config, 'discount', 0.99),
        tau=getattr(config, 'tau', 0.005),
        actor_bc_coef=getattr(config, 'actor_bc_coef', 1.0),  # 使用actor_bc_coef替代eta
        ema_decay=getattr(config, 'ema_decay', 0.995),
        step_start_ema=getattr(config, 'step_start_ema', 1000),
        update_ema_every=getattr(config, 'update_ema_every', 5),
        grad_clip=getattr(config, 'clip_grad_norm', 1.0),
        rank=rank,
        world_size=world_size,
        actor_update_freq=getattr(config, 'actor_update_freq', 1),
    )
    
    if config.load_model_path:
        # 加载模型参数
        model_state_dict = torch.load(config.load_model_path, map_location=device)
        model.load_state_dict(model_state_dict)
        print(f"模型参数已从 {config.load_model_path} 加载")
        
    return model


# 添加便捷的创建函数
def create_ddql_pointer(config, node_dim, rank=0, world_size=1):
    """创建使用Pointer Network模式的DDQL模型"""
    # 创建配置副本避免修改原配置
    config_copy = TrainConfig(**{k: v for k, v in vars(config).items() if k != 'diffusion_mode'})
    config_copy.diffusion_mode = "pointer"
    # 手动触发配置更新
    config_copy.__post_init__()
    return initialize_ddql(config_copy, node_dim, rank, world_size)


def create_ddql_simple(config, node_dim, rank=0, world_size=1):
    """创建使用Simple模式的DDQL模型"""
    # 创建配置副本避免修改原配置
    config_copy = TrainConfig(**{k: v for k, v in vars(config).items() if k != 'diffusion_mode'})
    config_copy.diffusion_mode = "simple"
    # 手动触发配置更新
    config_copy.__post_init__()
    return initialize_ddql(config_copy, node_dim, rank, world_size)