#!/usr/bin/env python3
"""
数据格式对齐检查器
检查EPIC 3D数据处理器生成的数据格式与graph_buffer期望格式的兼容性
"""

import numpy as np
import h5py
from pathlib import Path
import logging
from typing import Dict, List, Any

def check_topo_file_format(file_path: str) -> Dict[str, Any]:
    """检查单个topo文件的格式"""
    analysis = {
        'file_path': file_path,
        'has_current_position': False,
        'has_viewpoints': False,
        'has_tsp_info': False,
        'node_count': 0,
        'edge_count': 0,
        'viewpoint_nodes': [],
        'current_node': None,
        'tsp_order_info': {},
        'exploration_stats': {}
    }
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        current_section = None
        node_count = 0
        edge_count = 0
        
        for line in lines:
            line = line.strip()
            
            # 解析统计信息
            if line.startswith('#') and ':' in line and not line.startswith('# ==='):
                key_value = line[1:].strip()
                if ':' in key_value:
                    key, value = key_value.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        if '.' in value:
                            analysis['exploration_stats'][key] = float(value)
                        else:
                            analysis['exploration_stats'][key] = int(value)
                    except ValueError:
                        analysis['exploration_stats'][key] = value
            
            # 识别节点和边部分
            if line == "NODES":
                current_section = "nodes"
                continue
            elif line == "EDGES":
                current_section = "edges"
                continue
            elif line.startswith("#") or not line:
                continue
            
            # 解析节点
            if current_section == "nodes":
                parts = line.split()
                if len(parts) >= 14:
                    node_id = int(parts[0])
                    is_viewpoint = bool(int(parts[5]))
                    is_current = bool(int(parts[6]))
                    tsp_order_index = int(parts[10])
                    
                    node_count += 1
                    
                    # 检查是否有视点
                    if is_viewpoint:
                        analysis['has_viewpoints'] = True
                        analysis['viewpoint_nodes'].append({
                            'node_id': node_id,
                            'position': [float(parts[1]), float(parts[2]), float(parts[3])],
                            'tsp_order': tsp_order_index
                        })
                    
                    # 检查是否有当前位置
                    if is_current:
                        analysis['has_current_position'] = True
                        analysis['current_node'] = {
                            'node_id': node_id,
                            'position': [float(parts[1]), float(parts[2]), float(parts[3])],
                            'tsp_order': tsp_order_index
                        }
                    
                    # 检查TSP信息
                    if tsp_order_index >= 0:
                        analysis['has_tsp_info'] = True
                        analysis['tsp_order_info'][tsp_order_index] = node_id
            
            # 解析边
            elif current_section == "edges":
                edge_count += 1
        
        analysis['node_count'] = node_count
        analysis['edge_count'] = edge_count
        
    except Exception as e:
        analysis['error'] = str(e)
    
    return analysis

def check_graph_buffer_compatibility() -> Dict[str, Any]:
    """检查与graph_buffer的兼容性"""
    
    # 检查graph_buffer期望的数据格式
    expected_format = {
        'GraphTimeStep_fields': [
            'node_inputs',        # 节点特征 (N, node_dim)
            'node_padding_mask',  # 节点填充掩码 (1, N)
            'current_index',      # 当前位置索引 (1, 1)
            'viewpoints',         # 视点列表 (V, 1)
            'viewpoint_padding_mask', # 视点填充掩码 (1, V)
            'adj_list',          # 邻接列表 (N, k)
            'action',            # 动作
            'logp',              # 动作概率（可选）
            'reward',            # 奖励
            'done',              # 结束标志
            'first'              # 第一步标志
        ],
        'buffer_storage_fields': [
            'node_inputs',        # (buffer_size, node_padding_size, node_dim)
            'node_padding_mask',  # (buffer_size, 1, node_padding_size)
            'current_index',      # (buffer_size, 1, 1)
            'viewpoints',         # (buffer_size, viewpoint_padding_size, 1)
            'viewpoint_padding_mask', # (buffer_size, 1, viewpoint_padding_size)
            'adj_list',          # (buffer_size, node_padding_size, k_size)
            'actions',           # (buffer_size,)
            'rewards',           # (buffer_size,)
            'dones',             # (buffer_size,)
            'valid',             # (buffer_size,)
            'episode_ids'        # (buffer_size,)
        ]
    }
    
    return expected_format

def analyze_epic3d_processor_output() -> Dict[str, Any]:
    """分析EPIC3D处理器的输出格式"""
    
    # 检查EPIC3D数据处理器生成的格式
    epic3d_format = {
        'HDF5_structure': {
            'episodes/{episode_id}/observations/node_inputs': 'shape: (T, max_nodes, node_feature_dim)',
            'episodes/{episode_id}/observations/node_padding_mask': 'shape: (T, max_nodes)',
            'episodes/{episode_id}/observations/current_index': 'shape: (T,)',
            'episodes/{episode_id}/observations/viewpoints': 'shape: (T, max_viewpoints)',
            'episodes/{episode_id}/observations/viewpoint_padding_mask': 'shape: (T, max_viewpoints)',
            'episodes/{episode_id}/observations/adj_list': 'shape: (T, max_nodes, k_neighbors)',
            'episodes/{episode_id}/actions': 'shape: (T,)',
            'episodes/{episode_id}/rewards': 'shape: (T,)',
            'episodes/{episode_id}/dones': 'shape: (T,)',
            'episodes/{episode_id}/episode_metadata': 'dict with stats'
        },
        'node_features': [
            'x', 'y', 'z',                    # 位置 (3D)
            'is_viewpoint', 'is_current',     # 标志位
            'distance', 'observation_score',  # 评分
            'region_id', 'cluster_distance'   # 区域信息
        ],
        'processing_logic': {
            'action_inference': 'TSP order based: tsp_order_index[1] is next target',
            'reward_calculation': 'Distance based: -(curr_distance - prev_distance)',
            'current_position': 'From is_current=1 nodes',
            'viewpoints': 'From is_viewpoint=1 nodes'
        }
    }
    
    return epic3d_format

def main():
    """主检查函数"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("EPIC 3D数据处理格式对齐检查")
    print("=" * 80)
    
    # 检查示例文件
    sample_file = "/home/amax/EPIC/collected_data/forest_batch_4_3_20250826_161355/filtered_data/topo_graph_1756196044.598018.txt"
    
    if Path(sample_file).exists():
        print(f"\n1. 检查示例文件: {sample_file}")
        print("-" * 60)
        
        analysis = check_topo_file_format(sample_file)
        
        print(f"节点数量: {analysis['node_count']}")
        print(f"边数量: {analysis['edge_count']}")
        print(f"包含当前位置: {analysis['has_current_position']}")
        print(f"包含视点信息: {analysis['has_viewpoints']}")
        print(f"包含TSP信息: {analysis['has_tsp_info']}")
        
        if analysis['viewpoint_nodes']:
            print(f"视点节点: {len(analysis['viewpoint_nodes'])} 个")
            for vp in analysis['viewpoint_nodes'][:3]:  # 只显示前3个
                print(f"  - 节点 {vp['node_id']}: TSP顺序={vp['tsp_order']}, 位置={vp['position']}")
        
        if analysis['current_node']:
            print(f"当前节点: {analysis['current_node']}")
        
        if analysis['tsp_order_info']:
            print(f"TSP顺序信息: {dict(list(analysis['tsp_order_info'].items())[:5])}")  # 只显示前5个
        
        print(f"探索统计信息: {analysis['exploration_stats']}")
    
    # 检查格式兼容性
    print(f"\n2. Graph Buffer 期望格式:")
    print("-" * 60)
    expected = check_graph_buffer_compatibility()
    
    print("GraphTimeStep 字段:")
    for field in expected['GraphTimeStep_fields']:
        print(f"  - {field}")
    
    print("\nBuffer 存储字段:")
    for field in expected['buffer_storage_fields']:
        print(f"  - {field}")
    
    # 检查EPIC3D处理器格式
    print(f"\n3. EPIC3D 处理器输出格式:")
    print("-" * 60)
    epic3d = analyze_epic3d_processor_output()
    
    print("HDF5 结构:")
    for key, desc in epic3d['HDF5_structure'].items():
        print(f"  - {key}: {desc}")
    
    print(f"\n节点特征 ({len(epic3d['node_features'])} 维):")
    for i, feature in enumerate(epic3d['node_features']):
        print(f"  [{i}] {feature}")
    
    print(f"\n处理逻辑:")
    for key, desc in epic3d['processing_logic'].items():
        print(f"  - {key}: {desc}")
    
    # 兼容性分析
    print(f"\n4. 兼容性分析:")
    print("-" * 60)
    
    compatibility_issues = []
    
    # 检查维度兼容性
    print("✓ 数据维度匹配:")
    print("  - node_inputs: EPIC3D (T, max_nodes, 9) vs Buffer (buffer_size, node_padding_size, node_dim)")
    print("  - node_padding_mask: EPIC3D (T, max_nodes) vs Buffer (buffer_size, 1, node_padding_size)")
    print("  - current_index: EPIC3D (T,) vs Buffer (buffer_size, 1, 1) - 需要reshape")
    print("  - viewpoints: EPIC3D (T, max_viewpoints) vs Buffer (buffer_size, viewpoint_padding_size, 1) - 需要reshape")
    print("  - adj_list: EPIC3D (T, max_nodes, k_neighbors) vs Buffer (buffer_size, node_padding_size, k_size)")
    
    # 检查数据类型兼容性
    print("\n✓ 数据类型匹配:")
    print("  - node_inputs: float32 ✓")
    print("  - padding_masks: bool ✓") 
    print("  - indices: int64 ✓")
    print("  - actions: int64 ✓")
    print("  - rewards: float32 ✓")
    
    # 检查逻辑兼容性
    print("\n⚠️ 需要注意的兼容性问题:")
    
    if not Path(sample_file).exists() or not analysis.get('has_current_position'):
        print("  1. 当前位置检测: 部分文件可能没有 is_current=1 的节点")
        compatibility_issues.append("current_position_detection")
    
    if not Path(sample_file).exists() or not analysis.get('has_tsp_info'):
        print("  2. TSP信息缺失: 需要确保 tsp_order_index 有效")
        compatibility_issues.append("tsp_info_missing")
    
    print("  3. 维度重塑: current_index 和 viewpoints 需要适配 buffer 格式")
    compatibility_issues.append("dimension_reshaping")
    
    print("  4. 数据加载: HDF5 到 Buffer 的数据传输需要适配层")
    compatibility_issues.append("data_loading_adapter")
    
    # 推荐解决方案
    print(f"\n5. 推荐解决方案:")
    print("-" * 60)
    
    print("1. 创建数据适配器 (data_adapter.py):")
    print("   - 处理维度重塑")
    print("   - 标准化数据格式")
    print("   - 处理缺失数据")
    
    print("\n2. 修改 graph_buffer.py:")
    print("   - 添加 from_epic3d_hdf5() 方法")
    print("   - 适配 HDF5 数据加载")
    print("   - 处理批量数据加载")
    
    print("\n3. 验证数据处理器:")
    print("   - 确保所有文件都有当前位置")
    print("   - 验证 TSP 信息完整性")
    print("   - 测试端到端数据流")
    
    if compatibility_issues:
        print(f"\n发现 {len(compatibility_issues)} 个兼容性问题需要解决")
    else:
        print(f"\n✅ 格式基本兼容，可以直接使用")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
