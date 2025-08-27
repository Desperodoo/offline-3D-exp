#!/usr/bin/env python3
"""
数据格式检查器 - 检查EPIC 3D数据处理生成的数据与graph_buffer期望格式的兼容性
"""

import numpy as np
import h5py
from pathlib import Path
import yaml
from collections import namedtuple

# 从graph_buffer.py中导入的数据结构定义
GraphTimeStep = namedtuple('GraphTimeStep', 
                          ['node_inputs', 'node_padding_mask', 'current_index',
                           'viewpoints', 'viewpoint_padding_mask', 'adj_list',
                           'action', 'logp', 'reward', 'done', 'first'])

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_epic3d_format(data_file):
    """检查EPIC 3D数据文件格式"""
    print(f"=== 检查EPIC 3D数据文件: {data_file} ===")
    
    if not Path(data_file).exists():
        print(f"❌ 文件不存在: {data_file}")
        return False
    
    try:
        with h5py.File(data_file, 'r') as f:
            print("📊 数据集字段:")
            for key in f.keys():
                dataset = f[key]
                if isinstance(dataset, h5py.Dataset):
                    print(f"  {key}: {dataset.shape} ({dataset.dtype})")
                else:
                    print(f"  {key}: {type(dataset)}")
            
            print("\n📋 元数据属性:")
            for key in f.attrs.keys():
                print(f"  {key}: {f.attrs[key]}")
            
            # 检查必要字段是否存在
            required_fields = [
                'node_inputs', 'node_padding_mask', 'current_index',
                'viewpoints', 'viewpoint_padding_mask', 'adj_list',
                'actions', 'rewards', 'dones'
            ]
            
            missing_fields = []
            for field in required_fields:
                if field not in f.keys():
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"\n❌ 缺失必要字段: {missing_fields}")
                return False
            else:
                print(f"\n✅ 所有必要字段都存在")
            
            # 检查数据维度一致性
            sample_count = len(f['actions'])
            print(f"\n📏 数据维度检查 (样本数: {sample_count}):")
            
            for field in required_fields:
                if field in f.keys():
                    expected_first_dim = sample_count
                    actual_first_dim = f[field].shape[0]
                    if expected_first_dim == actual_first_dim:
                        print(f"  ✅ {field}: {f[field].shape}")
                    else:
                        print(f"  ❌ {field}: {f[field].shape} (期望第一维度: {expected_first_dim})")
                        return False
            
            # 检查具体的数据格式要求
            print(f"\n🔍 格式详细检查:")
            
            # 检查node_padding_mask是否为布尔类型
            if f['node_padding_mask'].dtype != bool:
                print(f"  ⚠️  node_padding_mask应为bool类型，当前为: {f['node_padding_mask'].dtype}")
            else:
                print(f"  ✅ node_padding_mask类型正确: {f['node_padding_mask'].dtype}")
            
            # 检查viewpoint_padding_mask是否为布尔类型  
            if f['viewpoint_padding_mask'].dtype != bool:
                print(f"  ⚠️  viewpoint_padding_mask应为bool类型，当前为: {f['viewpoint_padding_mask'].dtype}")
            else:
                print(f"  ✅ viewpoint_padding_mask类型正确: {f['viewpoint_padding_mask'].dtype}")
            
            # 检查dones是否为布尔类型
            if f['dones'].dtype != bool:
                print(f"  ⚠️  dones应为bool类型，当前为: {f['dones'].dtype}")
            else:
                print(f"  ✅ dones类型正确: {f['dones'].dtype}")
            
            # 检查actions是否为整数类型
            if not np.issubdtype(f['actions'].dtype, np.integer):
                print(f"  ❌ actions应为整数类型，当前为: {f['actions'].dtype}")
                return False
            else:
                print(f"  ✅ actions类型正确: {f['actions'].dtype}")
            
            # 检查rewards是否为浮点类型
            if not np.issubdtype(f['rewards'].dtype, np.floating):
                print(f"  ❌ rewards应为浮点类型，当前为: {f['rewards'].dtype}")
                return False
            else:
                print(f"  ✅ rewards类型正确: {f['rewards'].dtype}")
            
            print(f"\n📈 数据统计:")
            print(f"  动作范围: {f['actions'][:].min()} - {f['actions'][:].max()}")
            print(f"  奖励范围: {f['rewards'][:].min():.3f} - {f['rewards'][:].max():.3f}")
            print(f"  奖励均值: {f['rewards'][:].mean():.3f}")
            print(f"  完成标志数量: {f['dones'][:].sum()}")
            
            return True
            
    except Exception as e:
        print(f"❌ 检查文件时出错: {e}")
        return False

def check_graph_buffer_compatibility():
    """检查与graph_buffer的兼容性"""
    print(f"\n=== 检查与GraphReplayBuffer的兼容性 ===")
    
    # 从graph_buffer.py检查期望的数据格式
    print("📋 GraphTimeStep期望的字段:")
    for field in GraphTimeStep._fields:
        print(f"  - {field}")
    
    print("\n🔄 字段映射关系:")
    mapping = {
        'node_inputs': 'node_inputs ✅',
        'node_padding_mask': 'node_padding_mask ✅', 
        'current_index': 'current_index ✅',
        'viewpoints': 'viewpoints ✅',
        'viewpoint_padding_mask': 'viewpoint_padding_mask ✅',
        'adj_list': 'adj_list ✅',
        'action': 'actions ✅ (复数形式)',
        'logp': '❌ 缺失 (日志概率)',
        'reward': 'rewards ✅ (复数形式)',
        'done': 'dones ✅ (复数形式)', 
        'first': '❌ 缺失 (首个时间步标志)'
    }
    
    for graph_field, epic_field in mapping.items():
        print(f"  {graph_field} -> {epic_field}")
    
    print(f"\n⚠️  注意事项:")
    print(f"  1. logp字段缺失 - 这对离线RL训练可能不是问题")
    print(f"  2. first字段缺失 - 可能需要从episode边界推导")
    print(f"  3. 字段名称略有差异（单数vs复数）")

def check_topo_data_format(topo_file):
    """检查原始topo数据格式"""
    print(f"\n=== 检查原始Topo数据格式: {topo_file} ===")
    
    if not Path(topo_file).exists():
        print(f"❌ 文件不存在: {topo_file}")
        return False
    
    try:
        with open(topo_file, 'r') as f:
            lines = f.readlines()
        
        print(f"📄 文件信息: {len(lines)} 行")
        
        # 查找关键部分
        sections_found = {}
        node_count = 0
        edge_count = 0
        viewpoint_count = 0
        current_position_count = 0
        tsp_section = False
        viewpoint_section = False
        
        for i, line in enumerate(lines[:100]):  # 只检查前100行
            line = line.strip()
            if "EXPLORATION STATISTICS" in line:
                sections_found['stats'] = i
            elif line == "NODES":
                sections_found['nodes'] = i
            elif line == "EDGES":
                sections_found['edges'] = i
            elif line == "TSP_ORDER":
                sections_found['tsp'] = i
                tsp_section = True
            elif line == "VIEWPOINTS":
                sections_found['viewpoints'] = i
                viewpoint_section = True
            elif "is_viewpoint" in line and not line.startswith("#"):
                # 解析节点行
                parts = line.split()
                if len(parts) >= 6:
                    is_viewpoint = bool(int(parts[5]))
                    is_current = bool(int(parts[6])) if len(parts) > 6 else False
                    tsp_order = int(parts[10]) if len(parts) > 10 else -1
                    
                    if is_viewpoint:
                        viewpoint_count += 1
                    if is_current:
                        current_position_count += 1
                    node_count += 1
        
        print(f"📊 发现的数据段:")
        for section, line_num in sections_found.items():
            print(f"  {section}: 第{line_num}行")
        
        print(f"\n📈 数据统计 (前100行解析):")
        print(f"  节点数: {node_count}")
        print(f"  视点数: {viewpoint_count}")
        print(f"  当前位置数: {current_position_count}")
        print(f"  包含TSP段: {tsp_section}")
        print(f"  包含VIEWPOINT段: {viewpoint_section}")
        
        # 检查节点格式是否正确
        expected_node_fields = [
            'node_id', 'x', 'y', 'z', 'yaw', 'is_viewpoint', 'is_current', 
            'is_history', 'region_id', 'is_reachable', 'tsp_order_index', 
            'distance', 'observation_score', 'cluster_distance'
        ]
        
        print(f"\n📋 期望的节点字段 ({len(expected_node_fields)}个):")
        for field in expected_node_fields:
            print(f"  - {field}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查topo文件时出错: {e}")
        return False

def main():
    """主函数"""
    print("🔍 EPIC 3D 数据格式检查器")
    print("=" * 50)
    
    # 检查配置文件
    config_path = "epic3d_rl_config.yaml"
    if Path(config_path).exists():
        print(f"✅ 找到配置文件: {config_path}")
        config = load_config(config_path)
        print(f"📋 配置参数: {config}")
    else:
        print(f"⚠️  配置文件不存在: {config_path}")
    
    # 检查是否有示例数据文件
    example_topo = "/home/amax/EPIC/collected_data/forest_batch_4_3_20250826_161355/filtered_data/topo_graph_1756196044.598018.txt"
    if Path(example_topo).exists():
        check_topo_data_format(example_topo)
    else:
        print(f"⚠️  示例topo文件不存在: {example_topo}")
    
    # 检查是否有生成的HDF5数据文件
    example_hdf5 = "epic3d_dataset.h5"
    if Path(example_hdf5).exists():
        check_epic3d_format(example_hdf5)
    else:
        print(f"⚠️  示例HDF5文件不存在: {example_hdf5}")
    
    # 检查兼容性
    check_graph_buffer_compatibility()
    
    print(f"\n" + "=" * 50)
    print("🏁 检查完成")

if __name__ == "__main__":
    main()
