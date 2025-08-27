#!/usr/bin/env python3
"""
EPIC 3D 数据适配器 - 将EPIC 3D HDF5数据转换为与现有graph_buffer兼容的格式
"""

import h5py
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging


class EPIC3DDataAdapter:
    """
    EPIC 3D数据适配器，负责：
    1. 将EPIC 3D处理器生成的HDF5格式转换为graph_buffer兼容格式
    2. 处理维度重塑和数据类型转换
    3. 生成与旧版data_collector相同格式的批次文件
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def convert_epic3d_to_buffer_format(
        self, 
        input_path: str, 
        output_dir: str,
        batch_prefix: str = "batch"
    ) -> List[str]:
        """
        将EPIC 3D HDF5文件转换为buffer兼容格式
        
        参数:
            input_path: EPIC 3D处理器输出的HDF5文件路径
            output_dir: 输出目录
            batch_prefix: 批次文件前缀
            
        返回:
            生成的批次文件路径列表
        """
        if not Path(input_path).exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        batch_files = []
        
        with h5py.File(input_path, 'r') as f:
            # 读取episode边界信息
            episode_boundaries = f.attrs['episode_boundaries']
            num_episodes = f.attrs['num_episodes']
            
            self.logger.info(f"处理 {num_episodes} 个episodes，总样本数: {f.attrs['total_samples']}")
            
            # 读取原始数据
            original_data = self._load_epic3d_data(f)
            
            # 按episode边界分批处理
            for episode_idx in range(num_episodes):
                start_idx = episode_boundaries[episode_idx]
                end_idx = episode_boundaries[episode_idx + 1]
                
                # 提取当前episode的数据
                episode_data = self._extract_episode_data(original_data, start_idx, end_idx)
                
                # 转换为buffer兼容格式
                buffer_data = self._convert_to_buffer_format(episode_data)
                
                # 保存为批次文件
                batch_file_path = os.path.join(output_dir, f"{batch_prefix}_{episode_idx+1}.h5")
                self._save_batch_file(batch_file_path, buffer_data)
                batch_files.append(batch_file_path)
                
                if (episode_idx + 1) % 10 == 0:
                    self.logger.info(f"已处理 {episode_idx + 1}/{num_episodes} 个episodes")
        
        self.logger.info(f"转换完成，生成 {len(batch_files)} 个批次文件")
        return batch_files
    
    def _load_epic3d_data(self, f: h5py.File) -> Dict[str, np.ndarray]:
        """从EPIC 3D HDF5文件加载数据"""
        return {
            'node_inputs': f['node_inputs'][:],
            'node_padding_mask': f['node_padding_mask'][:],
            'current_index': f['current_index'][:],
            'viewpoints': f['viewpoints'][:],
            'viewpoint_padding_mask': f['viewpoint_padding_mask'][:],
            'adj_list': f['adj_list'][:],
            'actions': f['actions'][:],
            'rewards': f['rewards'][:],
            'dones': f['dones'][:]
        }
    
    def _extract_episode_data(
        self, 
        original_data: Dict[str, np.ndarray], 
        start_idx: int, 
        end_idx: int
    ) -> Dict[str, np.ndarray]:
        """提取单个episode的数据"""
        return {
            key: data[start_idx:end_idx] 
            for key, data in original_data.items()
        }
    
    def _convert_to_buffer_format(self, episode_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        转换为buffer兼容格式，处理维度重塑问题
        
        根据调试结果，需要处理的维度问题：
        - current_index: EPIC3D (T,) vs Buffer (buffer_size, 1, 1) 
        - viewpoints: EPIC3D (T, max_viewpoints) vs Buffer (buffer_size, viewpoint_padding_size, 1)
        - node_padding_mask: EPIC3D (T, max_nodes) vs Buffer (buffer_size, 1, node_padding_size)
        """
        T = len(episode_data['actions'])  # episode长度
        
        # 处理current_index维度重塑: (T,) → (T, 1, 1)
        current_index = episode_data['current_index'].reshape(T, 1, 1)
        
        # 处理viewpoints维度重塑: (T, max_viewpoints) → (T, max_viewpoints, 1)
        viewpoints = episode_data['viewpoints']
        if len(viewpoints.shape) == 2:
            viewpoints = viewpoints.reshape(T, viewpoints.shape[1], 1)
        
        # 处理node_padding_mask维度重塑: (T, max_nodes) → (T, 1, max_nodes)
        node_padding_mask = episode_data['node_padding_mask']
        if len(node_padding_mask.shape) == 2:
            node_padding_mask = node_padding_mask.reshape(T, 1, node_padding_mask.shape[1])
        
        # 处理viewpoint_padding_mask维度重塑: (T, max_viewpoints) → (T, 1, max_viewpoints)  
        viewpoint_padding_mask = episode_data['viewpoint_padding_mask']
        if len(viewpoint_padding_mask.shape) == 2:
            viewpoint_padding_mask = viewpoint_padding_mask.reshape(T, 1, viewpoint_padding_mask.shape[1])
        
        # 构建buffer格式数据
        buffer_data = {
            # 图结构数据 - 保持原维度
            'node_inputs': episode_data['node_inputs'].astype(np.float32),
            'adj_list': episode_data['adj_list'].astype(np.int64),
            
            # 重塑后的数据
            'node_padding_mask': node_padding_mask.astype(bool),
            'current_index': current_index.astype(np.int64),
            'viewpoints': viewpoints.astype(np.float32),
            'viewpoint_padding_mask': viewpoint_padding_mask.astype(bool),
            
            # RL数据 - 确保数据类型正确
            'actions': episode_data['actions'].astype(np.int64),
            'rewards': episode_data['rewards'].astype(np.float32), 
            'dones': episode_data['dones'].astype(bool),
        }
        
        return buffer_data
    
    def _save_batch_file(self, file_path: str, buffer_data: Dict[str, np.ndarray]):
        """
        保存批次文件，格式与旧版data_collector完全兼容
        """
        with h5py.File(file_path, 'w') as f:
            # 保存所有数据字段
            for key, data in buffer_data.items():
                f.create_dataset(key, data=data, compression='gzip', compression_opts=6)
            
            # 保存元数据 - 与旧版格式兼容
            f.attrs['episodes'] = 1  # 每个批次文件包含一个episode
            f.attrs['samples'] = len(buffer_data['actions'])
            
            # 保存维度信息
            if 'node_inputs' in buffer_data:
                f.attrs['node_dim'] = buffer_data['node_inputs'].shape[-1]
                f.attrs['num_nodes'] = buffer_data['node_inputs'].shape[1]
            
            # 添加兼容性标记
            f.attrs['format_version'] = 'epic3d_adapted'
            f.attrs['compatible_with'] = 'graph_buffer_v1'


def convert_epic3d_dataset(
    epic3d_file: str,
    output_dir: str,
    batch_prefix: str = "batch"
) -> List[str]:
    """
    便捷函数：将EPIC 3D数据集转换为buffer兼容格式
    
    参数:
        epic3d_file: EPIC 3D处理器输出的HDF5文件
        output_dir: 输出目录
        batch_prefix: 批次文件前缀
        
    返回:
        生成的批次文件路径列表
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 创建适配器并转换
    adapter = EPIC3DDataAdapter(logger)
    batch_files = adapter.convert_epic3d_to_buffer_format(
        input_path=epic3d_file,
        output_dir=output_dir,
        batch_prefix=batch_prefix
    )
    
    return batch_files


if __name__ == "__main__":
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EPIC 3D数据格式适配器')
    parser.add_argument('input', help='EPIC 3D HDF5文件路径')
    parser.add_argument('output_dir', help='输出目录')
    parser.add_argument('--batch_prefix', default='batch', help='批次文件前缀')
    
    args = parser.parse_args()
    
    batch_files = convert_epic3d_dataset(
        epic3d_file=args.input,
        output_dir=args.output_dir,
        batch_prefix=args.batch_prefix
    )
    
    print(f"转换完成！生成了 {len(batch_files)} 个批次文件:")
    for i, file_path in enumerate(batch_files):
        print(f"  {i+1}: {file_path}")
