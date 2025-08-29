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
from graph_buffer import load_merged_batch_files
from utils import get_batch_files, kill_mps_server, set_seed, wandb_init
from agent.iql import initialize_iql  # 导入从iql.py剥离出的DDPIQL类
from agent.cql import initialize_cql  # 导入CQL初始化函数
from agent.awr import initialize_awr  # 导入AWR初始化函数
from agent.td3bc import initialize_td3bc  # 导入TD3BC初始化函数
from agent.ddql import initialize_ddql
from agent.bc import initialize_bc  # 导入BC初始化函数
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

def train_model_with_mixed_batches(model, config, rank, world_size):
    """
    使用多策略混合数据训练模型，每个batch包含不同策略的数据
    
    参数:
        model: DDPIQL模型实例
        config: 训练配置
        rank: 当前进程的rank
        world_size: 总进程数
    """
    # 设置基本参数
    total_steps = model.total_it
    total_time = 0
    max_timesteps = config.max_timesteps
    epoches = config.epoches
        
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
            
            # # ===== 添加数据验证和调试信息 =====
            # if t <= 10 or (t + 1) % 100 == 0 or total_steps <= 20:  # 前10步、每100步或总步数少于20时打印调试信息
            #     if rank == 0:
            #         print(f"\n=== 调试信息 Step {t+1} ===")
            #         # 检查batch数据的基本统计信息
            #         for key, tensor in batch.items():
            #             if torch.is_tensor(tensor):
            #                 has_nan = torch.isnan(tensor).any().item()
            #                 has_inf = torch.isinf(tensor).any().item()
            #                 tensor_min = tensor.min().item() if tensor.numel() > 0 else 0
            #                 tensor_max = tensor.max().item() if tensor.numel() > 0 else 0
            #                 print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}, "
            #                       f"range=[{tensor_min:.4f}, {tensor_max:.4f}], "
            #                       f"has_nan={has_nan}, has_inf={has_inf}")
                            
            #                 # 特别检查padding_mask的值分布
            #                 if 'padding_mask' in key:
            #                     unique_vals = torch.unique(tensor)
            #                     print(f"    {key} unique values: {unique_vals.tolist()}")
                    
            #         # 检查节点数据的有效性
            #         node_inputs = batch['node_inputs']
            #         node_padding_mask = batch['node_padding_mask']
            #         B, N, F = node_inputs.shape
                    
            #         # 计算每个样本的有效节点数
            #         valid_nodes_per_sample = (~node_padding_mask.squeeze(1)).sum(dim=1)  # [B]
            #         print(f"  有效节点数统计: min={valid_nodes_per_sample.min().item()}, "
            #               f"max={valid_nodes_per_sample.max().item()}, "
            #               f"mean={valid_nodes_per_sample.float().mean().item():.1f}")
                    
            #         # 检查viewpoints数据
            #         viewpoint_padding_mask = batch['viewpoint_padding_mask']
            #         valid_vps_per_sample = (~viewpoint_padding_mask.squeeze(1)).sum(dim=1)
            #         print(f"  有效视点数统计: min={valid_vps_per_sample.min().item()}, "
            #               f"max={valid_vps_per_sample.max().item()}, "
            #               f"mean={valid_vps_per_sample.float().mean().item():.1f}")
                    
            #         # 检查动作是否在有效范围内
            #         actions = batch['actions']
            #         max_valid_vps = valid_vps_per_sample.max().item()
            #         invalid_actions = (actions >= valid_vps_per_sample).sum().item()
            #         print(f"  动作范围: min={actions.min().item()}, max={actions.max().item()}, "
            #               f"invalid_actions={invalid_actions}/{actions.numel()}")
                    
            #         print("========================")
                
            # 训练一步
            loss_dict = model.train(batch, config.mini_batch_size)
            
            # # 检查损失值是否正常
            # for loss_name, loss_value in loss_dict.items():
            #     if isinstance(loss_value, (int, float)):
            #         if loss_value != loss_value:  # NaN check
            #             raise ValueError(f"损失 {loss_name} 为 NaN: {loss_value}")
            #         if abs(loss_value) > 1e6:  # 异常大的损失值
            #             if rank == 0:
            #                 print(f"⚠️  警告: {loss_name} 异常大: {loss_value}")
            
            # 更新步数
            total_steps += 1
            
            # 定期更新超参数
            progress_remaining = 1.0
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
            
            # 定期保存检查点
            if rank == 0 and config.checkpoints_path and total_steps % config.save_freq == 0:
                save_path = os.path.join(config.checkpoints_path, f"checkpoint_{total_steps}.pt")
                save_model(model, save_path, rank=rank, message=f"步骤 {total_steps} 检查点")

    
        # 同步等待所有进程完成当前轮次
        if world_size > 1:
            dist.barrier()
    
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

def train_model(rank, world_size, config):
    """
    分布式训练模型的入口函数（支持IQL和DDQL）
    """
    # 设置随机种子（每个进程使用不同的种子）
    set_seed(config.seed + rank)
    
    # 设置分布式环境
    setup_ddp(rank, world_size)
    
    try:
        # 确保保存路径存在，仅在主进程创建目录
        if rank == 0 and config.checkpoints_path:
            os.makedirs(config.checkpoints_path, exist_ok=True)
            with open(os.path.join(config.checkpoints_path, "config.json"), "w") as f:
                json.dump(asdict(config), f, indent=4)
            
            # 主进程初始化Ray（如果尚未初始化）
            if not ray.is_initialized():
                ray.init()
        
        # 等待所有进程同步
        dist.barrier()
        
        # 初始化wandb（仅在主进程，且检查环境变量）
        if rank == 0 and config.project:
            wandb_init(asdict(config))
        
        # 从第一个批次文件提取维度信息
        print(f"进程 {rank}: 加载第一个批次以获取维度信息...")
        print(f"进程 {rank}: 数据维度: 节点特征={config.node_dim}, 节点数={config.num_nodes}, "
                f"视点数={config.max_viewpoints}, K近邻={config.k_size}")
        # 根据算法选择初始化方法
        print(f"进程 {rank}: 初始化DDP {config.algorithm.upper()}模型...")
        if config.algorithm.lower() == "iql":
            model = initialize_iql(
                config, 
                rank,
                world_size,
            )
        elif config.algorithm.lower() == "cql":
            model = initialize_cql(
                config, 
                rank,
                world_size,
            )
        elif config.algorithm.lower() == "awr":
            model = initialize_awr(
                config, 
                rank,
                world_size,
            )
        elif config.algorithm.lower() == "td3bc":
            model = initialize_td3bc(
                config, 
                rank,
                world_size,
            )
        elif config.algorithm.lower() == "ddql":
            model = initialize_ddql(
                config, 
                rank,
                world_size,
            )
        elif config.algorithm.lower() == "bc":
            model = initialize_bc(
                config, 
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
        )

        # 确保所有进程完成训练
        dist.barrier()
        
        # 清理分布式环境
        cleanup_ddp()
        
    except Exception as e:
        print(f"进程 {rank} 发生错误: {e}")
        import traceback
        traceback.print_exc()
        # 确保在出错时清理分布式环境
        if dist.is_initialized():
            cleanup_ddp()
        raise

def spawn_training(config, world_size):
    """
    使用torch.multiprocessing启动多进程训练
    """
    import torch.multiprocessing as mp
    
    # 设置多进程启动方式
    mp.spawn(
        train_model,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

def train_model_ddp(config):
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

    if world_size > 1:
        # 多GPU模式：使用DDP
        spawn_training(config, world_size)
    else:
        # 单GPU模式：直接在主进程中训练
        train_model(0, 1, config)
    
    # 关闭Ray（如果已初始化）
    if ray.is_initialized():
        ray.shutdown()
    
    print("多GPU训练完成！")
