#!/usr/bin/env python3
"""
EPIC 3D数据处理真实数据测试

使用真实的dungeon exploration数据测试EPIC3D数据处理系统
"""

import os
import sys
import yaml
import logging
import time
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline.epic3d_data_processor import EPIC3DDatasetBuilder

def setup_logging():
    """设置详细日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_real_data_processing():
    """测试真实数据处理"""
    logger = setup_logging()
    logger.info("开始真实数据处理测试...")
    
    # 1. 加载配置
    config_path = os.path.join(os.path.dirname(__file__), 'epic3d_rl_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✓ 配置加载完成")
    
    # 2. 设置测试数据路径
    data_dir = "/home/amax/EPIC/collected_data/dungeon_batch_1_0_20250827_153343"
    output_dir = "/home/amax/EPIC/src/offline/test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"测试数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    
    # 3. 验证数据目录存在
    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        return False
        
    filtered_data_dir = os.path.join(data_dir, 'filtered_data')
    if not os.path.exists(filtered_data_dir):
        logger.error(f"filtered_data目录不存在: {filtered_data_dir}")
        return False
    
    # 检查topo_graph文件数量
    import glob
    topo_files = glob.glob(os.path.join(filtered_data_dir, 'topo_graph_*.txt'))
    logger.info(f"找到 {len(topo_files)} 个topo_graph文件")
    
    if len(topo_files) == 0:
        logger.error("未找到topo_graph文件")
        return False
    
    # 4. 创建数据处理器
    builder = EPIC3DDatasetBuilder(config)
    logger.info("✓ 数据处理器创建成功")
    
    # 5. 处理数据
    try:
        start_time = time.time()
        
        # 单个episode测试
        logger.info(f"开始处理episode: {data_dir}")
        output_path = os.path.join(output_dir, "dungeon_test_episode.h5")
        
        # 调用处理方法
        result_path = builder.build_dataset_from_directories([data_dir], output_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"✓ 数据处理完成！")
        logger.info(f"处理时间: {processing_time:.2f}秒")
        logger.info(f"输出文件: {result_path}")
        
        # 6. 验证输出文件
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB
            logger.info(f"✓ 输出文件已生成，大小: {file_size:.2f}MB")
            
            # 检查批次文件
            base_path = result_path.replace('.h5', '')
            batch_files = glob.glob(f"{base_path}_batch_*.h5")
            logger.info(f"✓ 生成了 {len(batch_files)} 个批次文件")
            for batch_file in batch_files:
                batch_size = os.path.getsize(batch_file) / (1024 * 1024)
                logger.info(f"  - {os.path.basename(batch_file)}: {batch_size:.2f}MB")
                
        else:
            logger.error("✗ 输出文件未生成")
            return False
            
        # 7. 验证HDF5文件内容
        try:
            import h5py
            logger.info("验证HDF5文件内容...")
            
            with h5py.File(result_path, 'r') as f:
                logger.info("主要数据集:")
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        logger.info(f"  - {key}: {f[key].shape} ({f[key].dtype})")
                
                logger.info("文件属性:")
                for attr_name in f.attrs:
                    logger.info(f"  - {attr_name}: {f.attrs[attr_name]}")
            
            # 验证批次文件
            if batch_files:
                batch_file = batch_files[0]
                logger.info(f"验证批次文件: {os.path.basename(batch_file)}")
                with h5py.File(batch_file, 'r') as f:
                    logger.info("批次文件数据集:")
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            logger.info(f"  - {key}: {f[key].shape} ({f[key].dtype})")
                    
                    logger.info("批次文件属性:")
                    for attr_name in f.attrs:
                        logger.info(f"  - {attr_name}: {f.attrs[attr_name]}")
                        
        except Exception as e:
            logger.error(f"验证HDF5文件时出错: {e}")
            return False
        
        logger.info("✅ 真实数据处理测试成功完成！")
        return True
        
    except Exception as e:
        logger.error(f"✗ 数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_episodes():
    """测试多个episode处理"""
    logger = logging.getLogger(__name__)
    logger.info("开始多episode处理测试...")
    
    # 选择几个dungeon episode进行测试
    collected_data_dir = "/home/amax/EPIC/collected_data"
    import glob
    
    dungeon_episodes = glob.glob(os.path.join(collected_data_dir, "dungeon_batch_*"))
    dungeon_episodes.sort()
    
    # 选择前3个episode进行测试
    test_episodes = dungeon_episodes[:3]
    logger.info(f"选择 {len(test_episodes)} 个episode进行测试:")
    for ep in test_episodes:
        logger.info(f"  - {os.path.basename(ep)}")
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), 'epic3d_rl_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建处理器
    builder = EPIC3DDatasetBuilder(config)
    
    # 输出路径
    output_dir = "/home/amax/EPIC/src/offline/test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dungeon_multi_episodes.h5")
    
    try:
        start_time = time.time()
        result_path = builder.build_dataset_from_directories(test_episodes, output_path)
        end_time = time.time()
        
        logger.info(f"✓ 多episode处理完成！")
        logger.info(f"处理时间: {end_time - start_time:.2f}秒")
        logger.info(f"输出文件: {result_path}")
        
        # 验证输出
        import h5py
        with h5py.File(result_path, 'r') as f:
            logger.info(f"总episodes: {f.attrs.get('num_episodes', 'unknown')}")
            logger.info(f"总样本数: {f.attrs.get('total_samples', 'unknown')}")
            
            if 'actions' in f:
                logger.info(f"Actions shape: {f['actions'].shape}")
            if 'rewards' in f:
                logger.info(f"Rewards shape: {f['rewards'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"多episode处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("EPIC 3D 真实数据处理测试")
    print("=" * 60)
    
    # 测试单个episode
    print("\n1. 测试单个episode处理...")
    success1 = test_real_data_processing()
    
    if success1:
        print("\n2. 测试多个episode处理...")
        success2 = test_multiple_episodes()
    else:
        success2 = False
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 所有测试通过！真实数据处理系统工作正常。")
    elif success1:
        print("⚠️  单episode测试通过，多episode测试失败。")
    else:
        print("❌ 测试失败，需要检查数据处理逻辑。")
    print("=" * 60)
