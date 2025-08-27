#!/usr/bin/env python3
"""
调试统计数据解析
"""

import sys
import os
import logging

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline.epic3d_data_processor import EPIC3DDataParser

def debug_stats_parsing():
    """调试统计数据解析"""
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    parser = EPIC3DDataParser()
    
    # 测试两个连续的文件
    file1 = '/home/amax/EPIC/collected_data/dungeon_batch_1_0_20250827_153343/filtered_data/topo_graph_1756280032.883592.txt'
    file2 = '/home/amax/EPIC/collected_data/dungeon_batch_1_0_20250827_153343/filtered_data/topo_graph_1756280034.277884.txt'
    
    logger.info("=== 测试统计数据解析 ===")
    
    for i, file_path in enumerate([file1, file2], 1):
        logger.info(f"\n--- 文件 {i}: {os.path.basename(file_path)} ---")
        try:
            time_step = parser.parse_topo_file(file_path)
            logger.info(f"时间戳: {time_step.timestamp}")
            logger.info("探索统计数据:")
            for key, value in time_step.exploration_stats.items():
                logger.info(f"  {key}: {value}")
            
            # 重点检查我们关心的字段
            total_distance = time_step.exploration_stats.get('total_distance', 'NOT_FOUND')
            exploration_area = time_step.exploration_stats.get('exploration_area', 'NOT_FOUND')
            logger.info(f"关键字段:")
            logger.info(f"  total_distance: {total_distance}")
            logger.info(f"  exploration_area: {exploration_area}")
            
        except Exception as e:
            logger.error(f"解析失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 测试奖励计算
    logger.info("\n=== 测试奖励计算 ===")
    try:
        time_step1 = parser.parse_topo_file(file1)
        time_step2 = parser.parse_topo_file(file2)
        
        prev_stats = time_step1.exploration_stats
        curr_stats = time_step2.exploration_stats
        
        prev_distance = prev_stats.get('total_distance', 0.0)
        curr_distance = curr_stats.get('total_distance', 0.0)
        prev_area = prev_stats.get('exploration_area', 0.0)
        curr_area = curr_stats.get('exploration_area', 0.0)
        
        distance_change = curr_distance - prev_distance
        area_change = curr_area - prev_area
        
        logger.info(f"distance_change: {prev_distance:.6f} -> {curr_distance:.6f} = {distance_change:.6f}")
        logger.info(f"area_change: {prev_area:.6f} -> {curr_area:.6f} = {area_change:.6f}")
        
        area_reward = area_change * 0.1
        distance_penalty = -distance_change * 0.05
        total_reward = area_reward + distance_penalty
        
        logger.info(f"area_reward: {area_reward:.6f}")
        logger.info(f"distance_penalty: {distance_penalty:.6f}")
        logger.info(f"total_reward: {total_reward:.6f}")
        
    except Exception as e:
        logger.error(f"奖励计算测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_stats_parsing()
