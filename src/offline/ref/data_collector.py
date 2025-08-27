import os
import time
import numpy as np
import h5py
import ray
import json
from typing import List, Dict, Optional, Union, Any
from omegaconf import OmegaConf

from worker import Worker
from graph_buffer import convert_worker_buffer_to_samples
from iql_ddp import TrainConfig


class DataCollector:
    """
    离线RL数据收集器，用于从环境中收集数据并保存为图结构格式
    
    该类处理多Worker并行收集数据的逻辑，确保收集的数据与graph_buffer格式兼容
    """
    
    def __init__(
        self, 
        cfg, 
        save_path: str = './data',
        expert_model_path: Optional[str] = None,
        mode: str = 'expert',
        power: Optional[float] = 1.0
    ):
        """
        初始化数据收集器
        
        参数:
            cfg: 环境配置（OmegaConf对象）
            save_path: 数据保存路径
            expert_model_path: 专家模型路径，None表示使用默认PPO模型
        """
        self.cfg = cfg
        self.save_path = save_path
        self.expert_model_path = expert_model_path or 'checkpoint/PPO_FOREST_80_MEDIUM_DDP/checkpoint_24000.pth'
        
        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)
        # Save environment configuration
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_path = os.path.join(save_path, 'env_config.json')
        with open(cfg_path, 'w') as f:
            json.dump(cfg_dict, f, indent=4)
        print(f"Environment configuration saved to {cfg_path}")
        # 环境配置
        cfg_dict = OmegaConf.to_container(cfg)
        self.envs_per_worker = cfg_dict['runner']['num_envs_per_worker']
        
        # 性能跟踪
        self.collected_episodes = 0
        self.total_samples = 0
        self.all_perf_metrics = []
        self.mode = mode
        self.power = power

    def collect_data(
        self, 
        total_episodes: int, 
        num_workers: int = 2,
        max_workers: int = 120
    ) -> Dict[str, Any]:
        """
        从环境中收集数据
        
        参数:
            total_episodes: 需要收集的总episode数量
            num_workers: 使用的worker数量
            max_workers: 最大worker数限制
            
        返回:
            包含收集结果的字典，包括数据路径和性能指标
        """
        # 限制worker数量
        num_workers = min(num_workers, max_workers)
        
        # 计算需要多少轮收集
        episodes_per_round = num_workers * self.envs_per_worker
        num_rounds = (total_episodes + episodes_per_round - 1) // episodes_per_round
        
        print(f"计划收集 {total_episodes} 个episodes，使用 {num_workers} 个workers，"
              f"每个worker有 {self.envs_per_worker} 个环境，共需要 {num_rounds} 轮收集")
        
        # 创建数据目录
        timestamp = int(time.time())
        # data_dir = os.path.join(self.save_path, f'collected_data_{timestamp}')
        data_dir = self.save_path
        os.makedirs(data_dir, exist_ok=True)
        
        # 创建元数据
        metadata = {
            'timestamp': timestamp,
            'total_episodes': 0,
            'total_samples': 0,
            'batches': [],
            'performance': {}
        }
        
        # 初始化Workers
        algo_cfg = TrainConfig(
            device='cpu',  # 使用CPU进行数据收集
            seed=0,  # 固定随机种子
            gnn_hidden_dim=self.cfg.graph.embedding_dim,
            load_model_path=self.expert_model_path
        )
        
        workers = [Worker.remote(cfg=self.cfg, algo_cfg=algo_cfg, algo_type='iql', meta_agent_id=i, save_image=False) for i in range(num_workers)]
        
        print(f"使用专家策略: {self.expert_model_path}")
        
        try:
            # 初始化所有Workers
            init_refs = [worker.init.remote(None) for worker in workers]
            ray.get(init_refs)  # 等待所有worker初始化完成
            
            # 开始计时
            start_time = time.time()
            print(f"开始并行收集数据，保存到 {data_dir}...")
            
            # 重置统计信息
            self.collected_episodes = 0
            self.total_samples = 0
            self.all_perf_metrics = []
            
            # 循环收集数据
            for round_idx in range(num_rounds):
                # 计算本轮需要收集的episode数量
                episodes_this_round = min(episodes_per_round, 
                                        total_episodes - self.collected_episodes)
                if episodes_this_round <= 0:
                    break
                    
                print(f"\n----- 数据收集轮次 {round_idx+1}/{num_rounds} -----")
                print(f"本轮计划收集 {episodes_this_round} 个episodes")
                
                # 重新初始化worker（避免潜在的内存泄漏）
                init_refs = [worker.init.remote(state_dict=None) for worker in workers]
                ray.get(init_refs)  # 等待所有worker初始化完成
                
                # 执行单轮数据收集
                batch_info = self._collect_single_round(
                    workers=workers,
                    round_idx=round_idx,
                    data_dir=data_dir,
                    num_rounds=num_rounds
                )
                
                if batch_info:
                    metadata['batches'].append(batch_info)
                
                # 判断是否已达到目标
                if self.collected_episodes >= total_episodes:
                    break
                    
            # 更新元数据
            metadata['total_episodes'] = self.collected_episodes
            metadata['total_samples'] = self.total_samples
            metadata['collection_time'] = time.time() - start_time
            
            # 计算性能指标
            if self.all_perf_metrics:
                metadata['performance'] = self._calculate_performance_metrics()
                
            # 保存元数据
            metadata_path = os.path.join(data_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            print(f"\n所有数据收集完成，共 {self.collected_episodes} 个episodes，{self.total_samples} 条样本")
            print(f"数据已保存到 {data_dir} 目录中的 {len(metadata['batches'])} 个批次文件")
            print(f"元数据已保存到 {metadata_path}")
            
            return {
                'data_dir': data_dir,
                'metadata_path': metadata_path,
                'total_episodes': self.collected_episodes,
                'total_samples': self.total_samples,
                'performance': metadata['performance'] if 'performance' in metadata else {}
            }
            
        finally:
            # 确保关闭Workers
            for worker in workers:
                worker.close.remote()
                ray.kill(worker)
                
    def _collect_single_round(
        self,
        workers: List,
        round_idx: int,
        data_dir: str,
        num_rounds: int,
    ) -> Optional[Dict[str, Any]]:
        """
        收集单轮数据
        
        参数:
            workers: worker列表
            round_idx: 轮次索引
            data_dir: 数据目录
            
        返回:
            批次文件信息字典，如果收集失败则返回None
        """
        # 并行运行所有worker的rollout
        # Using an ease-out function to make percentage increase faster at beginning and slower at end
        t = (round_idx + 1) / num_rounds
        
        percentage = 0.5 + 0.5 * (1 - (1 - t) ** self.power)
        print(f"开始收集数据，模式：{self.mode}，进度: {percentage:.2%}")
        if self.mode == 'expert':
            # 使用专家策略
            rollout_refs = [worker.run_rollout.remote(percentage=1) for worker in workers]
        elif self.mode == 'random':
            # 使用随机策略
            rollout_refs = [worker.run_rollout.remote(percentage=0) for worker in workers]
        elif self.mode == 'replay':
            # 使用回放策略
            rollout_refs = [worker.run_rollout.remote(percentage=percentage) for worker in workers]
        else:
            raise ValueError(f"未知模式: {self.mode}")

        valid_workers = ray.get(rollout_refs)
        
        # 收集成功的worker数据
        rollout_data_list = [ray.get(worker.get_episode_buffer.remote()) 
                             for i, worker in enumerate(workers) if valid_workers[i]]
        perf_metrics_list = [ray.get(worker.get_perf_metrics.remote()) 
                              for i, worker in enumerate(workers) if valid_workers[i]]
        
        # 检查是否有数据收集成功
        if not rollout_data_list:
            print("警告: 本轮所有worker都没有成功收集数据，跳过本轮")
            return None
            
        # 将所有worker的数据转换为样本
        round_samples = []
        round_episodes = 0
        
        for worker_id, rollout_data in enumerate(rollout_data_list):
            # print(f"处理Worker {worker_id+1}的数据...")
            # 使用graph_buffer中的函数转换数据
            episode_samples = convert_worker_buffer_to_samples(rollout_data)
            round_samples.extend(episode_samples)
            
            successful_envs = perf_metrics_list[worker_id]['valid_envs']
            round_episodes += successful_envs
            # print(f"Worker {worker_id+1}: 成功收集了 {successful_envs} 个环境的数据")
            
        # 保存本轮收集到的数据到单独的批次文件
        if round_samples:
            batch_file_name = f'batch_{round_idx+1}.h5'
            batch_file_path = os.path.join(data_dir, batch_file_name)
            
            with h5py.File(batch_file_path, 'w') as f:
                sample_keys = round_samples[0].keys()
                for key in sample_keys:
                    field_data = [sample[key] for sample in round_samples]
                    if np.isscalar(field_data[0]) or isinstance(field_data[0], (bool, np.bool_)):
                        field_data = np.array(field_data)
                    else:
                        field_data = np.stack(field_data, axis=0)
                    
                    # 直接使用标准化的字段名，不需要转换
                    f.create_dataset(key, data=field_data)
                
                # 保存批次元数据
                f.attrs['episodes'] = round_episodes
                f.attrs['samples'] = len(round_samples)
                
                # 保存维度信息，以便后续加载时使用
                if 'node_inputs' in sample_keys:
                    # 直接从第一个样本获取维度信息
                    node_dim = round_samples[0]['node_inputs'].shape[-1]
                    num_nodes = round_samples[0]['node_inputs'].shape[0]
                    f.attrs['node_dim'] = node_dim
                    f.attrs['num_nodes'] = num_nodes
            
            # 更新统计信息
            self.collected_episodes += round_episodes
            self.total_samples += len(round_samples)
            self.all_perf_metrics.extend(perf_metrics_list)
            travel_dist = sum(metrics['travel_dist'] for metrics in perf_metrics_list) / len(perf_metrics_list)
            reward = sum(metrics['episode_reward'] for metrics in perf_metrics_list) / len(perf_metrics_list)
            print(f"本轮收集了 {round_episodes} 个episodes, {len(round_samples)} 条样本")
            print(f"平均旅行距离: {travel_dist:.2f}, 平均奖励: {reward:.2f}")
            print(f"总进度: {self.collected_episodes} episodes")
            
            # 构建批次信息
            batch_info = {
                'path': batch_file_path,
                'file_name': batch_file_name,
                'episodes': round_episodes,
                'samples': len(round_samples),
                'index': round_idx + 1
            }
            
            # 释放内存
            del round_samples
            import gc
            gc.collect()
            
            return batch_info
            
        return None
            
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """计算并返回性能指标"""
        if not self.all_perf_metrics:
            return {}
            
        perf_metrics = {
            'travel_dist': sum(metrics['travel_dist'] for metrics in self.all_perf_metrics),
            'success_rate': sum(metrics['success_rate'] for metrics in self.all_perf_metrics),
            'episode_reward': sum(metrics['episode_reward'] for metrics in self.all_perf_metrics),
            'valid_envs': sum(metrics['valid_envs'] for metrics in self.all_perf_metrics),
            'decision_steps': np.sum([metrics['decision_steps'] for metrics in self.all_perf_metrics]),
            'time_steps': np.sum([metrics['time_steps'] for metrics in self.all_perf_metrics]),
            'max_time_steps': max([metrics['max_time_steps'] for metrics in self.all_perf_metrics]),
            'max_num_nodes': max([metrics['max_num_nodes'] for metrics in self.all_perf_metrics]),
            'max_num_vps': max([metrics['max_num_vps'] for metrics in self.all_perf_metrics]),
            'max_k_size': max([metrics['max_k_size'] for metrics in self.all_perf_metrics]),
        }
        
        # 性能指标平均化
        if perf_metrics['valid_envs'] > 0:
            perf_metrics['travel_dist'] /= perf_metrics['valid_envs']
            perf_metrics['success_rate'] /= perf_metrics['valid_envs'] 
            perf_metrics['episode_reward'] /= perf_metrics['valid_envs']
            
        # 转换为普通的Python类型，确保可以正确序列化为JSON
        return {k: float(v) for k, v in perf_metrics.items() if isinstance(v, (int, float, np.number))}


def collect_training_data(
    cfg, 
    total_episodes: int,
    num_workers: int = 2,
    save_path: str = './results',
    expert_model_path: Optional[str] = None,
    mode: str = 'expert',
    power: float = 1.0
) -> Dict[str, Any]:
    """
    从环境收集训练数据的便捷函数
    
    参数:
        cfg: 环境配置
        total_episodes: 需要收集的总episode数量
        num_workers: 使用的worker数量
        save_path: 数据保存路径
        expert_model_path: 专家模型路径
        
    返回:
        收集结果字典
    """
    # 确保Ray已初始化
    if not ray.is_initialized():
        ray.init()
    
    # 创建收集器并收集数据
    collector = DataCollector(
        cfg=cfg, 
        save_path=save_path,
        expert_model_path=expert_model_path,
        mode=mode,
        power=power
    )
    
    # 收集并返回结果
    results = collector.collect_data(
        total_episodes=total_episodes,
        num_workers=num_workers
    )
    
    return results


if __name__ == "__main__":
    """直接运行此脚本可以独立收集数据"""
    import argparse
    import hydra
    from Exploration_Env.utils import preprocess_cfg
    
    parser = argparse.ArgumentParser(description='收集离线RL训练数据')
    
    # 数据收集参数
    parser.add_argument('--episodes', type=int, default=8, help='收集的episode数量')
    parser.add_argument('--num_workers', type=int, default=2, help='并行worker数量')
    parser.add_argument('--save_path', type=str, default='./data', help='数据保存路径')
    parser.add_argument('--expert_model_path', type=str, default=None, help='专家模型路径')
    
    # Hydra配置参数
    parser.add_argument('--config_path', type=str, default='Exploration_Env', help='配置路径')
    parser.add_argument('--config_name', type=str, default='config.yaml', help='配置文件名')
    
    args = parser.parse_args()
    
    # 初始化Ray
    if not ray.is_initialized():
        ray.init()
    
    try:
        # 加载环境配置
        with hydra.initialize_config_module(args.config_path):
            cfg = hydra.compose(config_name=args.config_name)
        
        # 处理环境配置
        cfg = preprocess_cfg(cfg)
        
        # 收集数据
        results = collect_training_data(
            cfg=cfg,
            total_episodes=args.episodes,
            num_workers=args.num_workers,
            save_path=args.save_path,
            expert_model_path=args.expert_model_path
        )
        
        print("数据收集完成!")
        print(f"数据目录: {results['data_dir']}")
        print(f"共收集 {results['total_episodes']} 个episodes，{results['total_samples']} 条样本")
        
    finally:
        # 确保关闭Ray
        if ray.is_initialized():
            ray.shutdown()
