#!/usr/bin/env python3
"""
测试修复后的奖励计算
"""

import sys
import os
import logging
import yaml

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline.epic3d_data_processor import EPIC3DDatasetBuilder

def test_reward_calculation():
    """测试修复后的奖励计算"""
    # 设置日志级别为DEBUG以查看详细信息
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.info("=== 测试修复后的奖励计算 ===")
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), 'epic3d_rl_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建构建器
    builder = EPIC3DDatasetBuilder(config)
    
    # 测试数据目录
    data_dir = '/home/amax/EPIC/collected_data/dungeon_batch_1_0_20250827_153343'
    output_path = '/tmp/test_rewards.h5'
    
    logger.info(f"处理数据目录: {data_dir}")
    logger.info(f"输出文件: {output_path}")
    
    try:
        # 处理数据
        result_path = builder.build_dataset_from_directories([data_dir], output_path)
        logger.info(f"✅ 数据处理完成: {result_path}")
        
        # 检查生成的批次文件
        import h5py
        import numpy as np
        
        batch_files = [f for f in os.listdir('/tmp') if f.startswith('test_rewards_batch_') and f.endswith('.h5')]
        if batch_files:
            batch_file = f'/tmp/{batch_files[0]}'
            logger.info(f"检查批次文件: {batch_file}")
            
            with h5py.File(batch_file, 'r') as f:
                rewards = f['rewards'][:]
                logger.info(f"奖励序列形状: {rewards.shape}")
                logger.info(f"奖励统计:")
                logger.info(f"  平均值: {np.mean(rewards):.6f}")
                logger.info(f"  标准差: {np.std(rewards):.6f}")
                logger.info(f"  最小值: {np.min(rewards):.6f}")
                logger.info(f"  最大值: {np.max(rewards):.6f}")
                logger.info(f"  非零奖励数量: {np.count_nonzero(rewards)}")
                logger.info(f"  前10个奖励: {rewards[:10]}")
                
                if np.count_nonzero(rewards) > 0:
                    logger.info("✅ 奖励计算修复成功！")
                else:
                    logger.error("❌ 奖励仍然全为0")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reward_calculation()
