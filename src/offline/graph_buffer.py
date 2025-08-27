import abc
import numpy as np
import torch
import h5py
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import os
import time

# 定义与worker_vec兼容的图结构数据时间步
GraphTimeStep = namedtuple('GraphTimeStep', 
                          ['node_inputs', 'node_padding_mask', 'current_index',
                           'viewpoints', 'viewpoint_padding_mask', 'adj_list',
                           'action', 'logp', 'reward', 'done', 'first'])

class GraphReplayBuffer(abc.ABC):
    """图结构数据的抽象缓冲区基类"""
    
    @abc.abstractmethod
    def add(self, time_step):
        """添加一个时间步到缓冲区"""
        pass

    @abc.abstractmethod
    def sample(self, batch_size):
        """从缓冲区采样一批数据"""
        pass

    @abc.abstractmethod
    def __len__(self):
        """返回缓冲区中样本数量"""
        pass


class EfficientGraphReplayBuffer(GraphReplayBuffer):
    """
    高效的图结构数据缓冲区实现，与worker_vec中的buffer结构保持一致
    
    专为图结构数据设计的离线RL训练缓冲区，支持存储节点特征、边缘掩码等图数据
    """
    
    def __init__(self, 
                 buffer_size: int, 
                 batch_size: int,
                 node_dim: int,
                 node_padding_size: int,
                 viewpoint_padding_size: int,
                 k_size: int,
                 discount: float = 0.99,
                 nstep: int = 1,
                 device: str = "cpu"):
        """
        初始化缓冲区
        
        参数:
            buffer_size: 缓冲区大小
            batch_size: 批量大小
            node_dim: 节点特征维度
            node_padding_size: 节点填充大小
            viewpoint_padding_size: 视点填充大小
            k_size: 邻接矩阵k近邻大小
            discount: 折扣因子
            nstep: n步回报的n值
            device: 存储设备
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.node_dim = node_dim
        self.node_padding_size = node_padding_size
        self.viewpoint_padding_size = viewpoint_padding_size
        self.k_size = k_size
        self.discount = discount
        self.nstep = nstep
        self.device = device
        
        # 缓冲区指针和状态
        self.idx = 0  # 当前写入位置
        self.full = False  # 缓冲区是否已满
        self.episode_start_idx = 0  # 当前episode起始位置
        
        # 初始化缓冲区存储
        self._initialize_buffer()
        
        # 用于n步回报的折扣向量
        self.discount_vec = np.power(discount, np.arange(nstep))
        self.next_dis = discount ** nstep
        
    def _initialize_buffer(self):
        """初始化缓冲区存储空间，与worker_vec中的结构一致"""
        # 图结构数据存储
        self.node_inputs = np.zeros((self.buffer_size, self.node_padding_size, self.node_dim), dtype=np.float32)
        self.node_padding_mask = np.zeros((self.buffer_size, 1, self.node_padding_size), dtype=np.bool_)
        self.current_index = np.zeros((self.buffer_size, 1, 1), dtype=np.int64)
        self.viewpoints = np.zeros((self.buffer_size, self.viewpoint_padding_size, 1), dtype=np.float32)
        self.viewpoint_padding_mask = np.zeros((self.buffer_size, 1, self.viewpoint_padding_size), dtype=np.bool_)
        self.adj_list = np.zeros((self.buffer_size, self.node_padding_size, self.k_size), dtype=np.int64)
        
        # 动作、奖励和其他RL数据
        # self.logp = np.zeros(self.buffer_size, dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.int64)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.bool_)
        
        # 有效样本标志（用于采样）
        self.valid = np.zeros(self.buffer_size, dtype=np.bool_)
        
        # 存储每个样本所属的episode id，方便后续处理
        self.episode_ids = np.zeros(self.buffer_size, dtype=np.int32)

    def add(self, time_step: GraphTimeStep):
        """
        添加一个时间步到缓冲区
        
        参数:
            time_step: GraphTimeStep类型，包含当前状态和下一个状态等信息
        """
        # 处理first信号（新episode的开始）
        if time_step.first:
            self.episode_start_idx = self.idx
            
        # 存储状态信息
        self.node_inputs[self.idx] = time_step.node_inputs
        self.node_padding_mask[self.idx] = time_step.node_padding_mask
        self.current_index[self.idx] = time_step.current_index
        self.viewpoints[self.idx] = time_step.viewpoints
        self.viewpoint_padding_mask[self.idx] = time_step.viewpoint_padding_mask
        self.adj_list[self.idx] = time_step.adj_list
        
        # 存储动作、奖励等
        self.actions[self.idx] = time_step.action
        # self.logp[self.idx] = time_step.logp
        self.rewards[self.idx] = time_step.reward
        self.dones[self.idx] = time_step.done
        
        # 更新有效标志（跳过nstep个转换后开始标记为有效）
        ep_idx = self.idx - self.episode_start_idx
        if ep_idx >= self.nstep - 1:
            self.valid[(self.idx - self.nstep + 1) % self.buffer_size] = True
        
        # 更新指针
        self.idx = (self.idx + 1) % self.buffer_size
        if self.idx == 0:
            self.full = True
            
    def add_batch(self, time_steps: List[GraphTimeStep]):
        """批量添加多个时间步"""
        for time_step in time_steps:
            self.add(time_step)

    def sample(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        从缓冲区采样一批数据
        
        参数:
            batch_size: 可选，采样批量大小，默认使用初始化时的batch_size
            
        返回:
            Dict[str, torch.Tensor]: 包含所有采样的数据字段的字典
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # 从有效样本中随机选择索引
        if np.sum(self.valid) == 0:
            raise ValueError("No valid transitions in buffer yet")
            
        indices = np.random.choice(np.where(self.valid)[0], size=batch_size)
        
        # 采样当前状态数据
        node_inputs = torch.FloatTensor(self.node_inputs[indices]).to(self.device)
        node_padding_mask = torch.BoolTensor(self.node_padding_mask[indices]).to(self.device)
        current_index = torch.LongTensor(self.current_index[indices]).to(self.device)
        viewpoints = torch.FloatTensor(self.viewpoints[indices]).to(self.device)
        viewpoint_padding_mask = torch.BoolTensor(self.viewpoint_padding_mask[indices]).to(self.device)
        adj_list = torch.LongTensor(self.adj_list[indices]).to(self.device)
        
        # 采样动作和奖励数据
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        # logp = torch.FloatTensor(self.logp[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        dones = torch.BoolTensor(self.dones[indices]).to(self.device)
        
        # 采样下一个状态数据（考虑n步）
        next_indices = (indices + self.nstep) % self.buffer_size
        valid_next = ~self.dones[indices]  # 只在非终止状态采样下一个状态
        
        next_node_inputs = torch.FloatTensor(self.node_inputs[next_indices]).to(self.device)
        next_node_padding_mask = torch.BoolTensor(self.node_padding_mask[next_indices]).to(self.device)
        next_current_index = torch.LongTensor(self.current_index[next_indices]).to(self.device)
        next_viewpoints = torch.FloatTensor(self.viewpoints[next_indices]).to(self.device)
        next_viewpoint_padding_mask = torch.BoolTensor(self.viewpoint_padding_mask[next_indices]).to(self.device)
        next_adj_list = torch.LongTensor(self.adj_list[next_indices]).to(self.device)
        
        # 关键修改：添加next_actions支持，修复数据类型问题
        # 获取下一步的动作（用于离线RL中的数据集动作）
        next_actions = torch.LongTensor(self.actions[next_indices]).to(self.device)
        # 对于终止状态，next_actions设为-1或当前动作（避免使用无效动作）
        # 修复：将valid_next转换为张量以匹配torch.where的要求
        valid_next_tensor = torch.BoolTensor(valid_next).to(self.device)
        next_actions = torch.where(valid_next_tensor, next_actions, actions)
        
        # 构建与RL算法兼容的batch字典
        batch = {
            # 当前状态
            'node_inputs': node_inputs,
            'node_padding_mask': node_padding_mask,
            'current_index': current_index,
            'viewpoints': viewpoints,
            'viewpoint_padding_mask': viewpoint_padding_mask,
            'adj_list': adj_list,
            
            # 动作和奖励
            'actions': actions,
            # 'logp': logp,
            'rewards': rewards,
            'dones': dones,
            
            # 下一个状态
            'next_node_inputs': next_node_inputs,
            'next_node_padding_mask': next_node_padding_mask,
            'next_current_index': next_current_index,
            'next_viewpoints': next_viewpoints,
            'next_viewpoint_padding_mask': next_viewpoint_padding_mask,
            'next_adj_list': next_adj_list,
            
            # 关键新增：下一步动作（用于离线RL数据集动作）
            'next_actions': next_actions,
            
            # 有效掩码
            'valid_mask': torch.BoolTensor(valid_next).to(self.device)
        }
        
        return batch
        
    def __len__(self):
        """返回缓冲区中有效样本数量"""
        return int(np.sum(self.valid))

class MergedGraphReplayBuffer(EfficientGraphReplayBuffer):
    """
    高效的合并图结构数据缓冲区实现
    
    在初始化时就将多个数据源的样本合并到一个大的缓冲区中，
    避免在采样时进行频繁的子缓冲区采样和合并操作
    """
    
    @staticmethod
    def from_batch_files(batch_files: List[Union[str, Dict]], 
                        batch_size: int = 256, 
                        buffer_limit: Optional[int] = None,
                        device: str = "cuda",
                        verbose: bool = True) -> 'MergedGraphReplayBuffer':
        """
        从多个批次文件创建合并缓冲区
        
        参数:
            batch_files: 批次文件列表，可以是文件路径或包含'path'键的字典
            batch_size: 每次采样的批量大小
            buffer_limit: 限制最大样本数，None表示无限制
            device: 设备
            verbose: 是否打印详细信息
            
        返回:
            合并后的缓冲区
        """
        if not batch_files:
            raise ValueError("批次文件列表为空")
            
        start_time = time.time()
        if verbose:
            print(f"开始从 {len(batch_files)} 个文件合并数据到统一缓冲区...")

        # 计算所有文件的总样本数
        total_samples = 0
        file_sample_counts = []
        
        for i, batch_file in enumerate(batch_files):
            file_path = batch_file['path'] if isinstance(batch_file, dict) else batch_file
            file_name = batch_file['file_name'] if isinstance(batch_file, dict) else os.path.basename(file_path)
            
            with h5py.File(file_path, 'r') as f:
                sample_count = len(f['rewards'])
                file_sample_counts.append(sample_count)
                total_samples += sample_count
                # 获取数据维度
                node_dim = f['node_inputs'].shape[-1]
                node_padding_size = f['node_inputs'].shape[1]
                viewpoint_padding_size = f['viewpoints'].shape[1]
                k_size = f['adj_list'].shape[-1]
                discount = f.attrs.get('discount', 0.99)
                if verbose:
                    if i == 0:
                        print(f"数据维度: 节点特征={node_dim}, 节点数={node_padding_size}, "
                            f"视点数={viewpoint_padding_size}, K近邻={k_size}")

                    print(f"文件 {file_name}: {sample_count} 个样本")
        
        # 如果设置了缓冲区大小限制，裁剪样本数
        if buffer_limit and buffer_limit < total_samples:
            if verbose:
                print(f"由于缓冲区大小限制 ({buffer_limit}), 将只使用部分样本")
            total_samples = buffer_limit
        
        # 创建缓冲区实例
        buffer = MergedGraphReplayBuffer(
            buffer_size=total_samples,
            batch_size=batch_size,
            node_dim=node_dim,
            node_padding_size=node_padding_size,
            viewpoint_padding_size=viewpoint_padding_size,
            k_size=k_size,
            discount=discount,
            device=device
        )
        
        # 初始化合并存储
        buffer._initialize_merged_storage()
        
        # 加载和合并所有文件的数据
        samples_loaded = 0
        for i, batch_file in enumerate(batch_files):
            file_path = batch_file['path'] if isinstance(batch_file, dict) else batch_file
            file_name = batch_file['file_name'] if isinstance(batch_file, dict) else os.path.basename(file_path)
            
            # 计算要加载的样本数
            samples_to_load = min(file_sample_counts[i], total_samples - samples_loaded)
            
            if samples_to_load <= 0:
                if verbose:
                    print(f"跳过文件 {file_name}，已达到缓冲区大小限制")
                break
                
            if verbose:
                print(f"加载文件 {i+1}/{len(batch_files)}: {file_name}，提取 {samples_to_load}/{file_sample_counts[i]} 个样本")
            
            # 加载数据到缓冲区对应的位置
            buffer._load_file_to_buffer(file_path, samples_loaded, samples_to_load)
            samples_loaded += samples_to_load
            
            if samples_loaded >= total_samples:
                break
        
        # 标记合并缓冲区已满
        buffer.full = True
        buffer.idx = 0  # 重置索引
        
        # 处理有效样本标记 - 简化处理，假设所有样本有效
        buffer.valid = np.ones(total_samples, dtype=np.bool_)
        
        # # 设置episode边界
        # buffer._handle_episode_boundaries()
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"数据合并完成，总计加载 {samples_loaded} 个样本，耗时 {elapsed:.2f} 秒")
            print(f"合并缓冲区大小: {len(buffer)} 有效样本")
            
        return buffer
    
    def _initialize_merged_storage(self):
        """初始化合并后的存储空间，与普通缓冲区初始化相同，这里只是为了语义清晰"""
        self._initialize_buffer()
        
    def _load_file_to_buffer(self, file_path: str, start_idx: int, samples_to_load: int):
        """
        加载单个文件的数据到缓冲区的指定位置
        
        参数:
            file_path: 文件路径
            start_idx: 起始索引位置
            samples_to_load: 要加载的样本数
        """
        with h5py.File(file_path, 'r') as f:
            # 计算实际要加载的样本数（不超过文件中的样本数）
            real_samples_to_load = min(samples_to_load, len(f['rewards']))
            end_idx = start_idx + real_samples_to_load
            
            # 加载各个字段的数据
            self.node_inputs[start_idx:end_idx] = f['node_inputs'][:real_samples_to_load]
            self.node_padding_mask[start_idx:end_idx] = f['node_padding_mask'][:real_samples_to_load]
            self.current_index[start_idx:end_idx] = f['current_index'][:real_samples_to_load]
            self.viewpoints[start_idx:end_idx] = f['viewpoints'][:real_samples_to_load]
            self.viewpoint_padding_mask[start_idx:end_idx] = f['viewpoint_padding_mask'][:real_samples_to_load]
            self.adj_list[start_idx:end_idx] = f['adj_list'][:real_samples_to_load]
            
            # 加载动作、奖励等数据
            self.actions[start_idx:end_idx] = f['actions'][:real_samples_to_load]
            
            # # 处理logp（可能不存在）
            # if 'logp' in f:
            #     self.logp[start_idx:end_idx] = f['logp'][:real_samples_to_load]
            # else:
            #     self.logp[start_idx:end_idx] = 0.0
                
            self.rewards[start_idx:end_idx] = f['rewards'][:real_samples_to_load]
            self.dones[start_idx:end_idx] = f['dones'][:real_samples_to_load]
            
            # 为这批数据设置episode_ids，用于标记来源
            file_id = hash(file_path) % 10000  # 使用哈希值作为文件ID
            self.episode_ids[start_idx:end_idx] = file_id
    
    def _handle_episode_boundaries(self):
        """
        处理不同文件之间的episode边界
        确保在文件交界处正确设置episode边界，防止跨文件的错误n步回报计算
        """
        # 找出所有文件分界点
        file_boundaries = np.where(np.diff(self.episode_ids) != 0)[0]
        
        # 在文件边界处设置done标志，确保不会跨文件计算n步回报
        for idx in file_boundaries:
            self.dones[idx] = True
            
        # 找出所有done的位置
        done_indices = np.where(self.dones)[0]
        
        # 将每个episode之后的第一个样本标记为episode开始
        for idx in done_indices:
            next_idx = (idx + 1) % self.buffer_size
            if next_idx < self.buffer_size:  # 确保不超出缓冲区
                # 这是一个新episode的开始
                pass  # 在此可添加特殊处理逻辑
    
    def sample(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        从合并缓冲区采样一批数据 
        
        由于继承自EfficientGraphReplayBuffer，这里直接复用其采样逻辑
        采样效率更高，无需在采样时合并多个缓冲区的数据
        """
        # 直接使用父类的采样方法
        return super().sample(batch_size)

def load_merged_batch_files(batch_files: List, batch_size: int = 256, 
                            buffer_limit: Optional[int] = None,
                            device: str = "cuda",
                            verbose: bool = True) -> MergedGraphReplayBuffer:
    """
    从多个批次文件加载数据到合并缓冲区的便捷函数
    
    参数:
        batch_files: 批次文件列表，每个元素可以是路径字符串或包含path键的字典
        batch_size: 采样批量大小
        buffer_limit: 可选，限制最大样本数，None表示无限制
        device: 设备
        verbose: 是否打印详细信息
        
    返回:
        加载了所有数据的MergedGraphReplayBuffer实例
    """
    return MergedGraphReplayBuffer.from_batch_files(
        batch_files=batch_files,
        batch_size=batch_size,
        buffer_limit=buffer_limit,
        device=device,
        verbose=verbose
    )

def convert_worker_buffer_to_samples(episode_buffer):
    """
    将worker的episode_buffer转换为样本列表，便于存储
    
    参数:
        episode_buffer: Worker中的单个episode数据
        
    返回:
        样本列表，每个样本是一个字典
    """
    samples = []
    data_size = episode_buffer['node_inputs'].size(0)
    
    for i in range(data_size):
        sample = {
            'node_inputs': episode_buffer['node_inputs'][i].cpu().numpy(),
            'node_padding_mask': episode_buffer['node_padding_mask'][i].cpu().numpy(),
            'current_index': episode_buffer['current_index'][i].cpu().numpy(),
            'viewpoints': episode_buffer['viewpoints'][i].cpu().numpy(),
            'viewpoint_padding_mask': episode_buffer['viewpoint_padding_mask'][i].cpu().numpy(),
            'adj_list': episode_buffer['adj_list'][i].cpu().numpy(),
            # 直接使用统一的字段名
            'actions': episode_buffer['actions'][i].item() if episode_buffer['actions'].dim() == 1 else episode_buffer['actions'][i].cpu().numpy(),
            # 'logp': episode_buffer['logp'][i].item() if episode_buffer['logp'].dim() == 1 else episode_buffer['logp'][i].cpu().numpy(),
            'rewards': episode_buffer['rewards'][i].item() if episode_buffer['rewards'].dim() == 1 else episode_buffer['rewards'][i].cpu().numpy(),
            'dones': bool(episode_buffer['dones'][i].item()) if episode_buffer['dones'].dim() == 1 else bool(episode_buffer['dones'][i].cpu().numpy()),
        }
        samples.append(sample)
    
    # 打印第一个样本的各个字段维度
    # if samples:
    #     print("\n样本数据维度信息:")
    #     for key, value in samples[0].items():
    #         if isinstance(value, np.ndarray):
    #             print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    #         else:
    #             print(f"  {key}: type={type(value)}")
    
    return samples
