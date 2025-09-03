import os
import argparse
import time
import torch
import hydra
import ray
import glob

# 导入自定义模块
from utils import set_seed
from agent.iql import TrainConfig as Config_IQL
from agent.cql import TrainConfig as Config_CQL
from agent.awr import TrainConfig as Config_AWR
from agent.td3bc import TrainConfig as Config_TD3BC
from agent.ddql import TrainConfig as Config_DDQL
from agent.bc import TrainConfig as Config_BC
from trainer import train_model_ddp  # 导入多GPU训练函数


def get_args():
    parser = argparse.ArgumentParser(description='运行IQL离线RL训练')
    
    # 流程控制参数
    parser.add_argument('--train_from_data', action='store_true', help='从数据训练模式')
    parser.add_argument('--project', type=str, default='epic3d', help='项目名称')
    parser.add_argument('--run_id', type=str, default='epic3d', help='运行ID')

    # 数据参数
    parser.add_argument('--collect_episodes', type=int, default=8, help='收集数据的episode数量')
    parser.add_argument('--data_path', type=str, nargs='+', default='/home/amax/EPIC/datasets_v3', 
                       help='离线数据路径或数据目录，可以指定多个路径(以空格分隔)')
    parser.add_argument('--expert_model_path', type=str, default=None, 
                        help='专家模型路径，默认使用预设的PPO模型')
    parser.add_argument('--batch_limit', type=str, default=None, 
                        help='训练时使用的最大批次数量，可以是整数(所有数据集共享)或JSON格式的字符串{"medium/forest": 5, "small/office": 10}')
    parser.add_argument('--power', type=float, default=1.0, help='数据收集的功率参数')

    # 训练参数
    parser.add_argument('--world_size', type=int, default=None, help='DDP训练使用的GPU数量，默认使用所有可用GPU')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--max_timesteps', type=int, default=100000, help='最大训练步数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--mini_batch_size', type=int, default=256, help='小批次大小')
    parser.add_argument('--gnn_hidden_dim', type=int, default=64, help='GNN嵌入维度')
    parser.add_argument('--epoches', type=int, default=1, help='训练的epoch数量')
    parser.add_argument("--algorithm", type=str, default="iql", 
                        choices=["iql", "cql", "awr", "td3bc", "ddql", "bc"], 
                        help="使用的算法，可选：iql, cql, awr, td3bc, ddql, bc, dql")
    parser.add_argument('--load_model_path', type=str, default=None, help='加载的模型路径')
    parser.add_argument('--actor_update_freq', type=int, default=1, help='actor更新频率')

    # IQL和DDQL共用参数
    parser.add_argument('--actor_bc_coef', type=float, default=0.3, help='行为克隆损失权重(IQL和DDQL通用，DDQL推荐0.2-0.3)')

    # IQL特有参数
    parser.add_argument('--expectile_tau', type=float, default=0.8, help='IQL期望值参数')
    parser.add_argument('--temperature', type=float, default=1.0, help='IQL温度参数')
    parser.add_argument('--training_mode', type=str, default='iql', help='选择使用iql或者bc')

    # CQL特有参数
    parser.add_argument('--cql_alpha', type=float, default=5.0, help='CQL损失权重')
    parser.add_argument('--cql_min_q_weight', type=float, default=10.0, help='CQL最小Q值权重')
    parser.add_argument('--cql_temp', type=float, default=1.0, help='CQL温度参数')
    parser.add_argument('--cql_max_target_backup', action='store_true', help='是否使用最大目标备份')
    parser.add_argument('--cql_clip_diff_min', type=float, default=float('-inf'), help='CQL差值裁剪最小值')
    parser.add_argument('--cql_clip_diff_max', type=float, default=float('inf'), help='CQL差值裁剪最大值')
    
    # AWR特有参数
    parser.add_argument('--awr_temperature', type=float, default=3.0, help='AWR温度参数')
    parser.add_argument('--value_loss_type', type=str, default='mse', choices=['mse', 'huber'], help='Value损失类型')
    parser.add_argument('--advantage_type', type=str, default='q_minus_v', 
                        choices=['q_minus_v', 'gae', 'td_error'], help='Advantage计算方式')
    parser.add_argument('--weight_clipping', type=float, default=20.0, help='权重裁剪上限')
    
    # TD3BC特有参数
    parser.add_argument('--policy_noise', type=float, default=0.2, help='目标策略平滑化噪声')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='噪声裁剪范围')
    parser.add_argument('--alpha', type=float, default=2.5, help='BC权重自适应参数')
    parser.add_argument('--normalize_q', action='store_true', help='是否归一化Q值用于计算BC权重')

    # DDQL特有参数 (更新以匹配新的配置)
    parser.add_argument('--n_timesteps', type=int, default=10, help='DDQL扩散步数')
    parser.add_argument('--diffusion_mode', type=str, default='pointer', choices=['pointer', 'simple'], 
                        help='DDQL扩散模式：pointer (动态动作空间) 或 simple (固定动作空间)')
    parser.add_argument('--use_gumbel_softmax', action='store_true', help='是否使用Gumbel-Softmax采样')
    parser.add_argument('--gumbel_hard', action='store_true', help='Gumbel-Softmax是否使用直通估计器')
    parser.add_argument('--temperature_start', type=float, default=1.0, help='DDQL初始采样温度')
    parser.add_argument('--temperature_end', type=float, default=1.0, help='DDQL最终采样温度')
    parser.add_argument('--temperature_decay_steps', type=int, default=50000, help='DDQL温度衰减步数')
    
    # DDQL稳定化参数 (新增)
    parser.add_argument('--use_dataset_actions', action='store_true', default=True, 
                        help='是否使用数据集动作而非actor采样')
    parser.add_argument('--use_smooth_loss', action='store_true', default=True, 
                        help='是否使用平滑损失（Huber）')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='Huber损失的delta参数')
    
    # DQL特有参数（与DDQL共享的稳定化参数）
    parser.add_argument('--entropy_reg_weight', type=float, default=0.01, help='熵正则化权重（DQL和DDQL通用）')
    
    # Diffusion Policy特有参数（简化版，与DDQL共享大部分参数）
    parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'], 
                        help='beta调度方式')

    # 保留旧参数以向后兼容（但不再使用）
    parser.add_argument('--beta_start', type=float, default=1e-4, help='beta起始值（已弃用）')
    parser.add_argument('--beta_end', type=float, default=0.02, help='beta结束值（已弃用）')
    parser.add_argument('--action_embedding_dim', type=int, default=32, help='离散动作嵌入维度（已弃用）')
    parser.add_argument('--time_embedding_dim', type=int, default=32, help='时间嵌入维度（已弃用）')
    parser.add_argument('--denoising_net_hidden_dim', type=int, default=256, help='去噪网络隐藏层维度（已弃用）')
    parser.add_argument('--num_denoising_layers', type=int, default=3, help='去噪网络层数（已弃用）')
    parser.add_argument('--clip_sample', action='store_true', help='是否裁剪采样结果（已弃用）')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'l1'], help='损失类型（已弃用）')
    parser.add_argument('--prediction_type', type=str, default='epsilon', choices=['epsilon', 'sample'], 
                        help='预测类型：epsilon (噪声) 或 sample (样本)（已弃用）')
    
    # 其他参数
    parser.add_argument('--save_path', type=str, default='./results', help='结果保存路径')
    parser.add_argument('--eval_freq', type=int, default=200, help='checkpoint保存间隔')
    # Hydra配置文件路径
    parser.add_argument('--config_path', type=str, default='./', help='Hydra配置文件路径')
    parser.add_argument('--config_name', type=str, default='config.yaml', help='Hydra配置文件名')
    
    return parser.parse_args()

def process_data_paths(data_paths):
    """
    处理数据路径参数，支持通配符，将单一路径或路径列表统一处理为标准化的路径列表
    """
    processed_paths = []
    
    # 如果是单一字符串，将其转换为列表处理
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    
    # 处理每个路径，展开通配符
    for path in data_paths:
        # 处理通配符路径 (如 data/medium/*)
        if '*' in path:
            expanded_paths = glob.glob(path)
            if not expanded_paths:
                print(f"警告: 通配符路径 '{path}' 未匹配到任何目录")
            else:
                # 只添加目录
                expanded_dirs = [p for p in expanded_paths if os.path.isdir(p)]
                processed_paths.extend(expanded_dirs)
                print(f"通配符路径 '{path}' 匹配到 {len(expanded_dirs)} 个目录")
        else:
            # 检查路径是否存在
            if not os.path.exists(path):
                print(f"警告: 路径 '{path}' 不存在")
            elif not os.path.isdir(path):
                print(f"警告: '{path}' 不是目录")
            else:
                processed_paths.append(path)
    
    # 数据路径去重
    processed_paths = list(set(processed_paths))
    
    # 显示最终路径列表
    print(f"处理后的数据路径列表 ({len(processed_paths)}个):")
    for i, path in enumerate(processed_paths):
        print(f"  {i+1}. {path}")
    
    return processed_paths

def parse_batch_limit(batch_limit_str):
    """
    解析批次限制参数，可以是整数或JSON格式字符串
    
    返回:
        整数或字典
    """
    if batch_limit_str is None:
        return None
    
    try:
        # 首先尝试解析为整数
        return int(batch_limit_str)
    except ValueError:
        # 如果不是整数，尝试解析为JSON
        try:
            import json
            return json.loads(batch_limit_str)
        except json.JSONDecodeError:
            print(f"警告: 无法解析批次限制参数 '{batch_limit_str}'，将使用所有可用数据")
            return None

def main():
    # 解析命令行参数
    args = get_args()
    
    # 直接使用Hydra API加载配置，而不是使用装饰器
    config_dir = os.path.join(os.getcwd(), args.config_path)
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name=args.config_name)
    cfg = cfg.data_processing
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 初始化Ray (如果尚未初始化)
    if not ray.is_initialized():
        ray.init()
    
    try:
        print("\n===== 离线训练阶段 =====")
        
        if not args.data_path:
            print("错误: 未提供数据路径，请使用--data_path参数指定数据集路径")
            return
        
        # 处理数据路径，支持多路径和通配符
        processed_data_paths = process_data_paths(args.data_path)
        
        if not processed_data_paths:
            print("错误: 处理后没有有效的数据路径，请检查--data_path参数")
            return
        
        # 解析批次限制参数
        batch_limit = parse_batch_limit(args.batch_limit)
        if isinstance(batch_limit, dict):
            print("使用数据集特定的批次限制:")
            for dataset, limit in batch_limit.items():
                print(f"  {dataset}: {limit}个批次")
        elif batch_limit is not None:
            print(f"所有数据集共用批次限制: {batch_limit}个批次")
        
        # 设置训练配置
        if args.algorithm == "iql":
            config = Config_IQL(
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=args.seed,
                max_timesteps=args.max_timesteps,
                batch_size=args.batch_size,
                mini_batch_size=args.mini_batch_size,
                # 网络参数
                gnn_hidden_dim=args.gnn_hidden_dim,
                clip_grad_norm=5,
                actor_update_freq=args.actor_update_freq,
                # IQL特有参数
                expectile_tau=args.expectile_tau,
                temperature=args.temperature,
                actor_bc_coef=args.actor_bc_coef,  # 使用统一的参数名
                # 保存路径
                checkpoints_path=os.path.join(args.save_path, f'models_{int(time.time())}'),
                project="IQL-Graph" if args.project is None else args.project,
                name=args.run_id if args.run_id is not None else f"IQL-{time.strftime('%Y%m%d-%H%M%S')}",
            )
            # 添加算法标识符
            config.algorithm = 'iql'
        elif args.algorithm == "ddql":
            config = Config_DDQL(
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=args.seed,
                max_timesteps=args.max_timesteps,
                batch_size=args.batch_size,
                mini_batch_size=args.mini_batch_size,
                # 网络基本参数
                gnn_hidden_dim=args.gnn_hidden_dim,
                clip_grad_norm=5,
                actor_update_freq=args.actor_update_freq,
                # DDQL特有参数 (更新以匹配新配置)
                max_viewpoints=cfg.max_viewpoints,
                actor_bc_coef=args.actor_bc_coef,  # 行为克隆损失权重
                n_timesteps=args.n_timesteps,
                diffusion_mode=args.diffusion_mode,
                # 新增：采样配置
                use_gumbel_softmax=args.use_gumbel_softmax,
                gumbel_hard=args.gumbel_hard,
                temperature_start=args.temperature_start,
                temperature_end=args.temperature_end,
                temperature_decay_steps=args.temperature_decay_steps,
                # 新增：critic稳定化配置
                use_dataset_actions=args.use_dataset_actions,
                use_smooth_loss=args.use_smooth_loss,
                huber_delta=args.huber_delta,
                entropy_reg_weight=args.entropy_reg_weight,  # 熵正则化权重
                # 保存路径
                checkpoints_path=os.path.join(args.save_path, f'models_{int(time.time())}'),
                project="DDQL-Graph" if args.project is None else args.project,
                name=args.run_id if args.run_id is not None else f"DDQL-{time.strftime('%Y%m%d-%H%M%S')}",
            )
            # 添加算法标识符
            config.algorithm = "ddql"
        elif args.algorithm == "cql":
            config = Config_CQL(
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=args.seed,
                max_timesteps=args.max_timesteps,
                batch_size=args.batch_size,
                mini_batch_size=args.mini_batch_size,
                # 网络参数
                gnn_hidden_dim=args.gnn_hidden_dim,
                clip_grad_norm=5,
                actor_update_freq=args.actor_update_freq,
                # CQL特有参数
                cql_alpha=args.cql_alpha,
                cql_min_q_weight=args.cql_min_q_weight,
                cql_temp=args.cql_temp,
                cql_max_target_backup=args.cql_max_target_backup,
                cql_clip_diff_min=args.cql_clip_diff_min,
                cql_clip_diff_max=args.cql_clip_diff_max,
                # 保存路径
                checkpoints_path=os.path.join(args.save_path, f'models_{int(time.time())}'),
                project="CQL-Graph" if args.project is None else args.project,
                name=args.run_id if args.run_id is not None else f"CQL-{time.strftime('%Y%m%d-%H%M%S')}",
            )
            # 添加算法标识符
            config.algorithm = "cql"
        elif args.algorithm == "awr":
            config = Config_AWR(
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=args.seed,
                max_timesteps=args.max_timesteps,
                batch_size=args.batch_size,
                mini_batch_size=args.mini_batch_size,
                # 网络参数
                gnn_hidden_dim=args.gnn_hidden_dim,
                clip_grad_norm=5,
                actor_update_freq=args.actor_update_freq,
                # AWR特有参数
                temperature=args.awr_temperature,
                value_loss_type=args.value_loss_type,
                advantage_type=args.advantage_type,
                weight_clipping=args.weight_clipping,
                # 保存路径
                checkpoints_path=os.path.join(args.save_path, f'models_{int(time.time())}'),
                project="AWR-Graph" if args.project is None else args.project,
                name=args.run_id if args.run_id is not None else f"AWR-{time.strftime('%Y%m%d-%H%M%S')}",
            )
            # 添加算法标识符
            config.algorithm = "awr"
        elif args.algorithm == "td3bc":
            config = Config_TD3BC(
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=args.seed,
                max_timesteps=args.max_timesteps,
                batch_size=args.batch_size,
                mini_batch_size=args.mini_batch_size,
                # 网络参数
                gnn_hidden_dim=args.gnn_hidden_dim,
                clip_grad_norm=5,
                actor_update_freq=args.actor_update_freq,
                # TD3BC特有参数
                policy_noise=args.policy_noise,
                noise_clip=args.noise_clip,
                alpha=args.alpha,
                normalize_q=args.normalize_q,
                # 保存路径
                checkpoints_path=os.path.join(args.save_path, f'models_{int(time.time())}'),
                project="TD3BC-Graph" if args.project is None else args.project,
                name=args.run_id if args.run_id is not None else f"TD3BC-{time.strftime('%Y%m%d-%H%M%S')}",
            )
            # 添加算法标识符
            config.algorithm = "td3bc"
        elif args.algorithm == "bc":
            config = Config_BC(
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=args.seed,
                max_timesteps=args.max_timesteps,
                batch_size=args.batch_size,
                mini_batch_size=args.mini_batch_size,
                # 网络参数
                gnn_hidden_dim=args.gnn_hidden_dim,
                clip_grad_norm=5,
                # 保存路径
                checkpoints_path=os.path.join(args.save_path, f'models_{int(time.time())}'),
                project="BC-Graph" if args.project is None else args.project,
                name=args.run_id if args.run_id is not None else f"BC-{time.strftime('%Y%m%d-%H%M%S')}",
            )
            # 添加算法标识符
            config.algorithm = "bc"
        else:
            raise ValueError(f"不支持的算法: {args.algorithm}")
        
        # 添加加载模型路径
        if args.load_model_path:
            config.load_model_path = args.load_model_path
        
        # 处理node_dim - 使用node_feature_dim
        config.node_dim = cfg.node_feature_dim
        config.num_nodes = cfg.max_nodes
        config.max_viewpoints = cfg.max_viewpoints
        config.k_size = cfg.k_size
        
        # 添加数据路径到配置（使用处理后的路径列表）
        config.data_path = processed_data_paths
        
        # 添加批次限制到配置
        if batch_limit is not None:
            config.batch_limit = batch_limit
            
        # 添加DDP相关配置
        if args.world_size is not None:
            config.world_size = args.world_size
        
        # 添加训练参数
        config.epoches = args.epoches
        
        # 启动训练
        if args.world_size is None:
            args.world_size = torch.cuda.device_count()
        
        if args.world_size > 1:
            print(f"使用{config.algorithm.upper()}算法在{args.world_size}个GPU上训练")
        else:
            print(f"使用{config.algorithm.upper()}算法在单GPU上训练")
        
        # 启动分布式训练 - 使用原版接口
        train_model_ddp(config)
    
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise
    finally:
        # 确保程序结束时关闭Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    # 初始化Ray (如果尚未初始化)
    if not ray.is_initialized():
        ray.init()
        
    main()
