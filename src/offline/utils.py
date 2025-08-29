import json
import os
import time
import uuid
import numpy as np
import torch
import torch.distributed as dist
import random

import wandb

def get_batch_files(data_path, batch_limit=None, rank=0, world_size=1, random_allocation=True, random_seed=0):
    """
    从数据路径获取批次文件列表，并平均分配给不同进程
    支持单个数据目录或多个数据目录列表
    
    参数:
        data_path: 数据目录路径，可以是单个字符串路径或路径列表
        batch_limit: 使用的最大批次数量，可以是整数或字典{'medium/forest': 5, 'small/office': 10}形式
        rank: 当前进程rank
        world_size: 总进程数
        random_allocation: 是否使用随机分配策略，默认True
        
    返回:
        分配给当前进程的批次文件列表
    """
    all_batch_files = []
    
    # 将单个路径转换为列表以统一处理
    if isinstance(data_path, str):
        data_paths = [data_path]
    else:
        data_paths = data_path
    
    # 验证输入路径
    for path in data_paths:
        if not os.path.exists(path):
            raise ValueError(f"数据路径不存在: {path}")
        if not os.path.isdir(path):
            raise ValueError(f"数据路径必须是目录: {path}")
    
    # 从每个数据路径收集批次文件
    for path_idx, path in enumerate(data_paths):
        # 获取相对路径作为数据集标识符
        # 使用data_path中的最后两级目录作为标识符，如medium/forest
        rel_path = '/'.join(path.rstrip('/').split('/')[-2:])
        
        if rank == 0:
            print(f"处理数据目录 ({path_idx+1}/{len(data_paths)}): {path} [标识: {rel_path}]")
        
        path_batch_files = []
        metadata_path = os.path.join(path, 'metadata.json')
        
        # 尝试从元数据文件中获取批次文件信息
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                path_batch_files = metadata.get('batches', [])
                # 为每个批次文件添加数据集标识
                for batch in path_batch_files:
                    batch['dataset'] = rel_path
                path_batch_files.sort(key=lambda x: x['index'])
                if rank == 0:
                    print(f"  从元数据找到 {len(path_batch_files)} 个批次文件")
            except Exception as e:
                if rank == 0:
                    print(f"  读取元数据文件出错: {e}，将尝试扫描目录中的h5文件")
                path_batch_files = []
        
        # 如果元数据读取失败或没有元数据文件，扫描目录
        if not path_batch_files:
            if rank == 0:
                print(f"  扫描目录中的h5文件...")
            
            # 修复：只包含批次文件，过滤掉索引文件
            all_h5_files = [f for f in os.listdir(path) if f.endswith('.h5')]
            h5_files = []
            
            for f in all_h5_files:
                # 只包含批次文件格式：*_batch_*.h5 或 dataset_batch_*.h5
                if ('_batch_' in f and not f.startswith('dataset.h5')) or f.startswith('dataset_batch_'):
                    h5_files.append(f)
                elif rank == 0:
                    print(f"  跳过非批次文件: {f}")
            
            # 智能排序：优先处理dataset_batch_开头的文件，然后按数字排序
            def sort_key(filename):
                if filename.startswith('dataset_batch_'):
                    try:
                        # 提取数字部分进行排序
                        num = int(filename.replace('dataset_batch_', '').replace('.h5', ''))
                        return (0, num)  # dataset_batch文件优先，按数字排序
                    except:
                        return (0, float('inf'))
                else:
                    return (1, filename)  # 其他文件排在后面，按字母排序
            
            h5_files.sort(key=sort_key)
            
            for idx, h5_file in enumerate(h5_files):
                file_path = os.path.join(path, h5_file)
                batch_info = {
                    'path': file_path,
                    'file_name': h5_file,
                    'index': idx + 1,
                    'dataset': rel_path,  # 添加数据集标识
                    'is_epic3d_dataset': h5_file.startswith('dataset_batch_')  # 标记EPIC3D数据集
                }
                path_batch_files.append(batch_info)
            
            if rank == 0:
                print(f"  扫描到 {len(path_batch_files)} 个h5文件")
        
        # 应用特定数据集的批次限制
        if isinstance(batch_limit, dict) and rel_path in batch_limit:
            dataset_limit = batch_limit[rel_path]
            if dataset_limit < len(path_batch_files):
                path_batch_files = path_batch_files[:dataset_limit]
                if rank == 0:
                    print(f"  数据集 {rel_path} 使用前 {dataset_limit}/{len(path_batch_files)} 个批次")
        
        # 添加到总批次文件列表
        all_batch_files.extend(path_batch_files)
    
    # 确认所有文件都存在
    valid_batch_files = [batch for batch in all_batch_files if os.path.exists(batch['path'])]
    if not valid_batch_files:
        raise ValueError("没有找到可用的数据批次文件")

    # 为确保所有进程使用同样的随机种子
    # 设置种子为数据集文件名的哈希值，确保相同的数据集在不同运行中获得相同的分配
    seed_value = random_seed
    for batch in valid_batch_files:
        seed_value += hash(batch.get('file_name', '')) % 10000
    # 调整种子，使其与rank和world_size无关
    seed_value = seed_value % 1000000

    # 广播主进程的种子值
    if world_size > 1:
        seed_tensor = torch.tensor([seed_value], device=f"cuda:{rank}")
        dist.broadcast(seed_tensor, 0)
        seed_value = seed_tensor.item()

    # 设置随机种子
    random.seed(seed_value)

    if rank == 0:
        print(f"使用随机分配策略，种子值：{seed_value}")
            
    # 应用全局批次数限制
    if isinstance(batch_limit, int) and batch_limit < len(valid_batch_files):
        # Randomly select batch_limit files
        valid_batch_files = random.sample(valid_batch_files, batch_limit)
        if rank == 0:
            print(f"随机采样 {batch_limit}/{len(all_batch_files)} 个批次进行训练")
    
    # 为每个进程分配批次文件
    if world_size > 1:
        # 确保批次文件数量不少于进程数量
        if len(valid_batch_files) < world_size:
            raise ValueError(f"批次文件数量({len(valid_batch_files)})少于进程数量({world_size})，无法分配")
        
        # 统计每个数据集的文件数量（仅主进程）
        if rank == 0:
            dataset_distribution = {}
            for batch in valid_batch_files:
                dataset = batch.get('dataset', 'unknown')
                if dataset not in dataset_distribution:
                    dataset_distribution[dataset] = 0
                dataset_distribution[dataset] += 1
            
            print(f"数据集分布情况:")
            for dataset, count in dataset_distribution.items():
                print(f"  {dataset}: {count} 个文件")
        
        # 计算每个进程分配的文件数
        files_per_process = len(valid_batch_files) // world_size
        remainder = len(valid_batch_files) % world_size
        
        # 随机分配策略
        if random_allocation:
            # 随机打乱文件顺序
            random.shuffle(valid_batch_files)
            
            # 对打乱后的文件按数据集分类
            dataset_files = {}
            for batch in valid_batch_files:
                dataset = batch.get('dataset', 'unknown')
                if dataset not in dataset_files:
                    dataset_files[dataset] = []
                dataset_files[dataset].append(batch)
            
            # 通过交错分配不同数据集的文件，保证每个进程获得多样化的数据
            assigned_files = []
            datasets = list(dataset_files.keys())
            
            # 计算分配给当前进程的文件数（考虑余数）
            my_file_count = files_per_process + (1 if rank < remainder else 0)
            
            # 进行交错分配
            current_dataset_idx = 0
            while len(assigned_files) < my_file_count and datasets:
                # 循环使用不同数据集
                dataset = datasets[current_dataset_idx % len(datasets)]
                
                # 如果当前数据集还有文件
                if dataset_files[dataset]:
                    # 计算当前进程应该拿第几个文件
                    index = rank
                    # 如果文件数量够，直接分配
                    if index < len(dataset_files[dataset]):
                        assigned_files.append(dataset_files[dataset][index])
                        # 移除已分配的文件
                        dataset_files[dataset].pop(index)
                    
                # 移动到下一个数据集
                current_dataset_idx += 1
                
                # 如果已经循环完所有数据集一次，检查是否有空数据集
                if current_dataset_idx % len(datasets) == 0:
                    # 移除空数据集
                    datasets = [ds for ds in datasets if dataset_files[ds]]
                    
                    # 如果所有数据集都为空但还需要更多文件，使用剩余的文件
                    if not datasets and len(assigned_files) < my_file_count:
                        # 收集所有剩余文件
                        remaining_files = []
                        for remaining in valid_batch_files:
                            if remaining not in assigned_files:
                                remaining_files.append(remaining)
                        
                        # 计算当前进程应该从剩余文件中拿哪些
                        start_idx = len(assigned_files) + rank * (my_file_count - len(assigned_files))
                        end_idx = start_idx + (my_file_count - len(assigned_files))
                        
                        # 确保索引不越界
                        if start_idx < len(remaining_files):
                            assigned_files.extend(remaining_files[start_idx:min(end_idx, len(remaining_files))])
                        
                        break
            
        else:
            # 原始的顺序分配策略
            # 确保文件数能均分，舍弃多余的文件
            usable_files_count = (len(valid_batch_files) // world_size) * world_size
            if usable_files_count < len(valid_batch_files) and rank == 0:
                print(f"警告: {len(valid_batch_files) - usable_files_count} 个文件因不能均分而被舍弃")
            
            valid_batch_files = valid_batch_files[:usable_files_count]
            files_per_process = len(valid_batch_files) // world_size
            
            # 计算当前进程的文件分配范围
            start_idx = rank * files_per_process
            end_idx = start_idx + files_per_process
            
            # 分配文件
            assigned_files = valid_batch_files[start_idx:end_idx]
            
        # 统计当前进程分配的数据集分布
        local_distribution = {}
        for batch in assigned_files:
            dataset = batch.get('dataset', 'unknown')
            if dataset not in local_distribution:
                local_distribution[dataset] = 0
            local_distribution[dataset] += 1
        
        distribution_str = ", ".join([f"{ds}: {cnt}" for ds, cnt in local_distribution.items()])
        print(f"进程 {rank}: 分配了 {len(assigned_files)} 个批次文件 ({distribution_str})")
        
        return assigned_files
    else:
        # 单进程模式，使用所有文件
        if rank == 0:
            # 统计每个数据集的文件数量
            dataset_distribution = {}
            for batch in valid_batch_files:
                dataset = batch.get('dataset', 'unknown')
                if dataset not in dataset_distribution:
                    dataset_distribution[dataset] = 0
                dataset_distribution[dataset] += 1
            
            print(f"数据集分布情况:")
            for dataset, count in dataset_distribution.items():
                print(f"  {dataset}: {count} 个文件")
            
            print(f"使用全部 {len(valid_batch_files)} 个批次进行训练")
        
        return valid_batch_files
    
# 添加MPS服务器清理函数
def kill_mps_server():
    """
    安全关闭所有nvidia-cuda-mps-server进程
    """
    try:
        import subprocess
        import os
        import signal
        
        # 查找所有nvidia-cuda-mps-server进程
        result = subprocess.run(
            ["pgrep", "nvidia-cuda-mps"],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 如果找到进程
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"发现 {len(pids)} 个NVIDIA MPS服务器进程，正在终止...")
            
            # 尝试优雅地终止每个进程
            for pid in pids:
                if pid.strip():
                    try:
                        pid_int = int(pid.strip())
                        os.kill(pid_int, signal.SIGTERM)
                        print(f"发送SIGTERM信号到MPS进程 {pid_int}")
                    except (ValueError, ProcessLookupError) as e:
                        print(f"终止进程 {pid} 时出错: {e}")
            
            # 检查是否所有进程都已终止
            time.sleep(0.5)
            
            result = subprocess.run(
                ["pgrep", "nvidia-cuda-mps"],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0:
                remaining_pids = result.stdout.strip().split('\n')
                print(f"仍有 {len(remaining_pids)} 个MPS进程，强制终止...")
                
                for pid in remaining_pids:
                    if pid.strip():
                        try:
                            pid_int = int(pid.strip())
                            os.kill(pid_int, signal.SIGKILL)
                            print(f"发送SIGKILL信号到MPS进程 {pid_int}")
                        except (ValueError, ProcessLookupError) as e:
                            print(f"强制终止进程 {pid} 时出错: {e}")
    
    except Exception as e:
        print(f"尝试终止MPS服务器时发生错误: {e}")
        import traceback
        traceback.print_exc()
        
def set_seed(seed: int, deterministic_torch: bool = False):
    """设置随机种子以确保可重现性"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def wandb_init(config: dict) -> None:
    """初始化WandB"""
    # 检查wandb是否已经初始化
    if hasattr(wandb, 'run') and wandb.run is not None:
        print("wandb已经被初始化，使用现有实例")
        return

    # 确保所有必要的键都存在，使用默认值处理缺失情况
    wandb_config = {
        "project": config.get("project", "OfflineRL-Project"),
        "name": config.get("name", f"run-{str(uuid.uuid4())[:8]}"),
        # group是可选的
        "id": config.get("name", "run") + "-" + str(uuid.uuid4())[:8]
    }
    
    # 如果存在group，则添加
    if "group" in config:
        wandb_config["group"] = config["group"]
    
    # 初始化wandb
    wandb.init(
        config=config,
        **wandb_config
    )
    wandb.run.save()

