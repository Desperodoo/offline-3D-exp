import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import wandb
from dataclasses import asdict, dataclass
import time
import argparse
import json
import h5py
import gc
import copy
import random
import uuid
import ray
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

from graph_buffer import load_batch_data_to_buffer, load_merged_batch_files
from utils import get_batch_files, kill_mps_server, set_seed, wandb_init
from iql_ddp import initialize_iql  # 导入从iql_ddp.py剥离出的DDPIQL类
from cql_ddp import initialize_cql  # 导入CQL初始化函数
from awr_ddp import initialize_awr  # 导入AWR初始化函数
from td3bc_ddp import initialize_td3bc  # 导入TD3BC初始化函数
from ddql_graph_discrete import initialize_ddql
from bc_ddp import initialize_bc  # 导入BC初始化函数
from dql import initialize_dql  # 导入DQL初始化函数
# from diql_graph_discrete import initialize_ddql
from evaluator_remote import RemoteEvaluator  # 导入远程评估器

TensorBatch = List[torch.Tensor]
warnings.filterwarnings("ignore", category=DeprecationWarning)

# DDPIQL类已移动到iql_ddp.py

def save_model(model, save_path, rank=0, message=None):
    """
    保存模型到指定路径
    
    参数:
        model: DDPIQL模型实例
        save_path: 保存路径
        rank: 当前进程的rank（默认为0）
        message: 保存成功后的额外消息（可选）
    """
    model_state_dict = model.get_state_dict()
    torch.save(model_state_dict, save_path)
    if message:
        print(f"保存模型到 {save_path} - {message}")
    else:
        print(f"保存模型到 {save_path}")
    
    return model_state_dict

def setup_ddp(rank, world_size):
    """
    初始化分布式训练环境
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理分布式训练环境"""
    try:
        dist.destroy_process_group()
    finally:
        # 确保MPS服务器被关闭
        kill_mps_server()

def train_model_with_mixed_batches(model, config, rank, world_size, evaluator=None):
    """
    使用多策略混合数据训练模型，每个batch包含不同策略的数据
    
    参数:
        model: DDPIQL模型实例
        config: 训练配置
        rank: 当前进程的rank
        world_size: 总进程数
        evaluator: 远程评估器实例(仅rank=0进程使用)
    """
    # 设置基本参数
    total_steps = model.total_it
    total_time = 0
    max_timesteps = config.max_timesteps
    epoches = config.epoches
    
    # 获取评估开始步数配置
    eval_start_steps = getattr(config, 'eval_start', 0)
    
    # 启动持续评估任务（仅主进程且满足评估开始条件）
    evaluation_future = None
    if rank == 0 and evaluator is not None and total_steps >= eval_start_steps:
        print(f"启动持续评估过程（当前步数：{total_steps}，评估开始步数：{eval_start_steps}）...")
        model_state_dict = save_model(model, os.path.join(config.checkpoints_path, "initial_model.pt"), 
                                     rank=rank, message="初始模型")
        
        evaluation_future = evaluator.evaluate_model_dict.remote(
            model_dict=model_state_dict,
            primary_metric=getattr(config, 'primary_metric', 'travel_dist'),
            verbose=True
        )
    elif rank == 0 and evaluator is not None and total_steps < eval_start_steps:
        print(f"当前步数 {total_steps} 小于评估开始步数 {eval_start_steps}，暂不开始评估")
        
    # 启动训练
    for epoch in range(1, epoches + 1):
        if rank == 0:
            epoch_start = time.time()
            print(f"\n===== Epoch {epoch} =====")

        # 检查输入数据路径
        if not hasattr(config, 'data_path') or not config.data_path:
            raise ValueError("必须提供数据路径 (data_path)")
        
        # 获取批次文件列表 - 分配不同文件给不同进程
        batch_files = get_batch_files(
            data_path=config.data_path,
            batch_limit=getattr(config, 'batch_limit', None),
            rank=rank,
            world_size=world_size,
            random_seed=epoch
        )
        
        if not batch_files:
            print(f"进程 {rank}: 没有分配到批次文件，将等待其他进程完成")
            dist.barrier()
            return

        # ===== 关键修改: 一次性加载所有数据到混合缓冲区 =====
        print(f"进程 {rank}: 加载所有批次文件到混合缓冲区...")
        
        # 提取每个批次文件的相对路径作为权重依据（可根据需要自定义权重）
        # 默认所有文件权重相同，这里可以根据文件名或其他特征设置不同的权重
        buffer_weights = [1.0] * len(batch_files)
        
        # 将所有文件加载到一个混合缓冲区
        buffer = load_merged_batch_files(
            batch_files,
            batch_size=config.batch_size,
            device=f"cuda:{rank}",
            verbose=(rank == 0)  # 只在主进程打印详细信息
        )
    
        # 计算总训练步数
        total_train_steps = min(len(buffer), max_timesteps)
        if rank == 0:
            print(f"混合缓冲区总样本数: {len(buffer)}, 预计训练步数: {total_train_steps}")
    
        # 直接在混合缓冲区上训练指定步数
        for t in range(total_train_steps):
            # 从混合缓冲区采样批次数据 - 这里的batch会包含多种策略的数据
            batch = buffer.sample()
            
            # 训练一步
            loss_dict = model.train(batch, config.mini_batch_size)
            
            # 更新步数
            total_steps += 1
            
            # 检查评估结果（仅主进程且满足评估开始条件）
            if rank == 0 and evaluation_future is not None and total_steps >= eval_start_steps:
                ready_futures, remaining_futures = ray.wait([evaluation_future], timeout=0)
                
                if ready_futures:  # 评估完成
                    eval_metrics = ray.get(ready_futures[0])
                    
                    if eval_metrics:
                        if config.project:
                            wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=total_steps)
                        
                        print(f"步骤 {total_steps}: 开始新一轮持续评估...")
                        model_state_dict = model.get_state_dict()
                        # 重新评估模型
                        evaluation_future = evaluator.evaluate_model_dict.remote(
                            model_dict=model_state_dict,
                            primary_metric=getattr(config, 'primary_metric', 'travel_dist'),
                            verbose=True
                        )
            
            # 如果还未开始评估但已达到评估开始步数，则启动评估
            elif rank == 0 and evaluator is not None and evaluation_future is None and total_steps >= eval_start_steps:
                print(f"步骤 {total_steps}: 达到评估开始步数 {eval_start_steps}，启动首次评估...")
                model_state_dict = model.get_state_dict()
                evaluation_future = evaluator.evaluate_model_dict.remote(
                    model_dict=model_state_dict,
                    primary_metric=getattr(config, 'primary_metric', 'travel_dist'),
                    verbose=True
                )
            progress_remaining = 1.0
            
            # 定期更新超参数
            if total_steps % 2 == 0:
                progress_remaining = 1.0 - t / total_train_steps
                
                # 统一更新所有超参数（学习率、系数等）
                hyperparams_info = model.update_hyperparameters(progress_remaining)
                
                # 记录超参数变化 (如果使用wandb)
                if rank == 0 and config.project:
                    wandb.log(hyperparams_info, step=total_steps)
            
            # 记录训练日志（仅主进程）
            if rank == 0 and total_steps % 2 == 0:
                critic_loss = loss_dict.get('q_loss', 0)
                actor_loss = loss_dict.get('actor_loss', 0)
                value_loss = loss_dict.get('value_loss', 0)
                bc_loss = loss_dict.get('bc_loss', 0)
                print(f"步骤 {total_steps}/{total_train_steps * epoches}: "
                        f"critic_loss = {critic_loss:.4f}, "
                        f"actor_loss = {actor_loss:.4f}, "
                        f"value_loss = {value_loss:.4f}, "
                        f"bc_loss = {bc_loss:.4f}, "
                        f"use_dataset_actions = {loss_dict.get('use_dataset_actions', -1)}, "
                        f"using_dataset_actions = {loss_dict.get('using_dataset_actions', -1)}, "
                        f"use_smooth_loss = {loss_dict.get('use_smooth_loss', -1)}, "
                        f"progress = {(1-progress_remaining)*100:.1f}%")
                
                if config.project:
                    wandb.log(loss_dict, step=total_steps)
            
            # 定期保存检查点（仅在达到评估开始步数后）
            if rank == 0 and config.checkpoints_path and total_steps % config.eval_freq == 0 and total_steps >= eval_start_steps:
                save_path = os.path.join(config.checkpoints_path, f"checkpoint_{total_steps}.pt")
                save_model(model, save_path, rank=rank, message=f"步骤 {total_steps} 检查点")

    
        # 同步等待所有进程完成当前轮次
        if world_size > 1:
            dist.barrier()
    
        # 等待最后一次评估完成（如果有）
        if rank == 0 and evaluation_future is not None:
            print("等待最后一次评估完成...")
            try:
                eval_metrics = ray.get(evaluation_future)
                if eval_metrics:
                    print("最终评估结果:")
                    for k, v in eval_metrics.items():
                        if isinstance(v, (int, float, np.number)):
                            print(f"{k}: {v:.4f}")
            except Exception as e:
                print(f"获取最终评估结果时出错: {e}")
    
        # 清理资源
        del buffer
        gc.collect()

        if rank == 0:
            # 计算轮次时间（仅主进程）
            epoch_time = time.time() - epoch_start
            total_time += epoch_time
            print(f"Epoch {epoch} 完成，耗时 {epoch_time:.2f} 秒，总训练步数: {total_steps}")
        
    # 保存最终模型 (仅主进程)
    if rank == 0 and config.checkpoints_path:
        final_path = os.path.join(config.checkpoints_path, "final_model.pt")
        save_model(model, final_path, rank=rank, 
                  message=f"训练完成，总耗时 {total_time:.2f} 秒，总步数 {total_steps}")
    torch.cuda.empty_cache()
    
    # 同步所有进程
    if world_size > 1:
        dist.barrier()

def train_model(rank, world_size, env_cfg, config):
    """
    分布式训练模型的入口函数（支持IQL和DDQL）
    """
    # 设置随机种子（每个进程使用不同的种子）
    set_seed(config.seed + rank)
    
    # 设置分布式环境
    setup_ddp(rank, world_size)
    
    # 远程评估器（仅在主进程创建）
    evaluator = None
    
    try:
        # 确保保存路径存在，仅在主进程创建目录
        if rank == 0 and config.checkpoints_path:
            os.makedirs(config.checkpoints_path, exist_ok=True)
            with open(os.path.join(config.checkpoints_path, "config.json"), "w") as f:
                json.dump(asdict(config), f, indent=4)
            
            # 主进程初始化Ray（如果尚未初始化）
            if not ray.is_initialized():
                ray.init()
            
            # 创建远程评估器 - 传入配置和保存路径
            if getattr(config, 'enable_eval', True):
                print("创建远程评估器...")
                # 使用环境配置的num_envs_per_worker和num_workers来并行评估
                evaluator = RemoteEvaluator.remote(
                    env_cfg=env_cfg,
                    config=config,
                    num_workers=getattr(config, 'eval_workers', 2),
                    save_images=getattr(config, 'save_eval_images', False),
                    checkpoints_path=config.checkpoints_path,
                    device=getattr(config, 'eval_device', 'cpu'),
                )
                
                print(f"评估配置: workers={getattr(config, 'eval_workers', 2)}, "
                      f"envs_per_worker={env_cfg.runner.num_envs_per_worker}, " 
                      f"total_envs={getattr(config, 'eval_workers', 2) * env_cfg.runner.num_envs_per_worker}")
        
        # 等待所有进程同步
        dist.barrier()
        
        # 初始化wandb（仅在主进程，且检查环境变量）
        if rank == 0 and config.project:
            wandb_init(asdict(config))
        
        # 从第一个批次文件提取维度信息
        print(f"进程 {rank}: 加载第一个批次以获取维度信息...")
        
        # 提取维度信息
        node_dim = env_cfg.graph.node_input_dim
        node_padding_size = env_cfg.graph.node_padding_size
        viewpoint_padding_size = env_cfg.graph.viewpoint_padding_size
        k_size = env_cfg.graph.k_size
        
        # 更新配置
        config.node_dim = node_dim
        config.num_nodes = node_padding_size
        config.viewpoint_padding_size = viewpoint_padding_size
        print(f"进程 {rank}: 数据维度: 节点特征={node_dim}, 节点数={node_padding_size}, "
                f"视点数={viewpoint_padding_size}, K近邻={k_size}")
        
        # 确保所有进程获取的维度一致
        if world_size > 1:
            dims = torch.tensor([node_dim, node_padding_size, viewpoint_padding_size, k_size], 
                              device=f"cuda:{rank}")
            dist.broadcast(dims, 0)  # 以进程0的维度为准
            
            if rank > 0:
                node_dim, node_padding_size, viewpoint_padding_size, k_size = dims.tolist()
                config.node_dim = node_dim
                config.num_nodes = node_padding_size
        
        # 根据算法选择初始化方法
        print(f"进程 {rank}: 初始化DDP {config.algorithm.upper()}模型...")
        if config.algorithm.lower() == "iql":
            model = initialize_iql(
                config, 
                node_dim,
                rank,
                world_size,
            )
        elif config.algorithm.lower() == "cql":
            model = initialize_cql(
                config, 
                node_dim,
                rank,
                world_size,
            )
        elif config.algorithm.lower() == "awr":
            model = initialize_awr(
                config, 
                node_dim,
                rank,
                world_size,
            )
        elif config.algorithm.lower() == "td3bc":
            model = initialize_td3bc(
                config, 
                node_dim,
                rank,
                world_size,
            )
        elif config.algorithm.lower() == "ddql":
            model = initialize_ddql(
                config, 
                node_dim,
                rank,
                world_size,
            )
        elif config.algorithm.lower() == "bc":
            model = initialize_bc(
                config, 
                node_dim,
                rank,
                world_size,
            )
        elif config.algorithm.lower() == "dql":
            model = initialize_dql(
                config, 
                node_dim,
                rank,
                world_size,
            )
        else:
            raise ValueError(f"不支持的算法类型: {config.algorithm}")
        
        # 等待所有进程加载完成
        dist.barrier()

        # ===== 修改: 使用混合批次训练 =====
        train_model_with_mixed_batches(
            model=model, 
            config=config, 
            rank=rank, 
            world_size=world_size,
            evaluator=evaluator if rank == 0 else None
        )

        
        # 确保所有进程完成训练
        dist.barrier()
        
        # 关闭远程评估器（仅主进程）
        if rank == 0 and evaluator is not None:
            ray.get(evaluator.close.remote())
        
        # 清理分布式环境
        cleanup_ddp()
        
    except Exception as e:
        print(f"进程 {rank} 发生错误: {e}")
        import traceback
        traceback.print_exc()
        # 确保在出错时清理分布式环境
        if dist.is_initialized():
            cleanup_ddp()
        # 确保关闭评估器
        if rank == 0 and evaluator is not None:
            ray.get(evaluator.close.remote())
        raise

def spawn_training(env_cfg, config, world_size):
    """
    使用torch.multiprocessing启动多进程训练
    """
    import torch.multiprocessing as mp
    
    # 设置多进程启动方式
    mp.spawn(
        train_model,
        args=(world_size, env_cfg, config),
        nprocs=world_size,
        join=True
    )


def train_model_ddp(env_cfg, config):
    """
    多GPU训练模型的统一入口函数
    
    参数:
        env_cfg: 环境配置
        config: 训练配置
    
    返回:
        None
    """
    # 设置随机种子
    set_seed(config.seed)
    
    # 打印算法类型
    print(f"使用算法: {config.algorithm.upper()}")
    
    # 确定可用GPU数量
    num_gpus = torch.cuda.device_count()
    world_size = min(num_gpus, getattr(config, 'world_size', num_gpus))
    
    print(f"检测到 {num_gpus} 个GPU，使用 {world_size} 个GPU进行分布式训练")
    
    # 将评估相关配置添加到config
    if not hasattr(config, 'enable_eval'):
        config.enable_eval = True  # 默认启用评估
    if not hasattr(config, 'eval_episodes'):
        config.eval_episodes = 5  # 每次评估的episode数
    if not hasattr(config, 'eval_workers'):
        config.eval_workers = getattr(config, 'num_workers', 2)  # 评估使用的worker数量
    if not hasattr(config, 'primary_metric'):
        config.primary_metric = getattr(config, 'primary_metric', 'travel_dist')  # 主要评估指标
    
    if world_size > 1:
        # 多GPU模式：使用DDP
        spawn_training(env_cfg, config, world_size)
    else:
        # 单GPU模式：直接在主进程中训练
        train_model(0, 1, env_cfg, config)
    
    # 关闭Ray（如果已初始化）
    if ray.is_initialized():
        ray.shutdown()
    
    print("多GPU训练完成！")
