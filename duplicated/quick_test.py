#!/usr/bin/env python3
"""
快速测试改进后的数据采集脚本
"""

import sys
import os
sys.path.append("/home/amax/EPIC")

from data_collection_script import DataCollectionScript

def main():
    print("=== 快速测试数据采集脚本 ===")
    
    # 创建采集脚本实例（30秒测试）
    collector = DataCollectionScript(collection_time=30)
    
    # 测试配置选择
    print("1. 测试配置选择...")
    try:
        config = collector.select_random_config()
        print(f"✓ 选择配置: {config['map_type']} - {config['batch_id']}/{config['point_id']}")
        print(f"  初始位置: ({config['init_x']:.2f}, {config['init_y']:.2f})")
    except Exception as e:
        print(f"✗ 配置选择失败: {e}")
        return 1
    
    # 测试文件搜索功能
    print("\n2. 测试文件搜索...")
    try:
        collector.collection_start_time = 0  # 查找所有文件
        topo_files, stats_files = collector.find_collected_data()
        print(f"✓ 找到 {len(topo_files)} 个拓扑文件, {len(stats_files)} 个统计文件")
        
        if topo_files:
            print("  最新的拓扑文件:")
            for f in sorted(topo_files)[-3:]:  # 显示最新的3个
                print(f"    {f}")
    except Exception as e:
        print(f"✗ 文件搜索失败: {e}")
        return 1
    
    print("\n✓ 基础功能测试通过")
    print("可以运行完整的数据采集测试:")
    print("  python3 data_collection_script.py -t 30")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
