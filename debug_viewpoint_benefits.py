#!/usr/bin/env python3
"""
调试脚本：检查保存的topo graph数据中的视点收益信息
用于验证observation_score和cluster_distance是否正确保存
"""

import sys
import os
import glob
from typing import List, Tuple

def parse_node_line(line: str) -> dict:
    """解析节点行数据"""
    parts = line.strip().split()
    if len(parts) < 14:
        return None
    
    try:
        return {
            'node_id': int(parts[0]),
            'x': float(parts[1]),
            'y': float(parts[2]), 
            'z': float(parts[3]),
            'yaw': float(parts[4]),
            'is_viewpoint': bool(int(parts[5])),
            'is_current': bool(int(parts[6])),
            'is_history': bool(int(parts[7])),
            'region_id': int(parts[8]),
            'is_reachable': bool(int(parts[9])),
            'tsp_order_index': int(parts[10]),
            'distance': float(parts[11]),
            'observation_score': float(parts[12]),
            'cluster_distance': float(parts[13])
        }
    except (ValueError, IndexError) as e:
        print(f"Warning: Failed to parse line: {line.strip()}")
        print(f"Error: {e}")
        return None

def analyze_topo_graph_file(filepath: str) -> None:
    """分析单个topo graph文件"""
    print(f"\n{'='*60}")
    print(f"分析文件: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    nodes = []
    viewpoint_nodes = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # 找到NODES部分
        in_nodes_section = False
        for line in lines:
            line = line.strip()
            
            if line == "NODES":
                in_nodes_section = True
                continue
            elif line == "EDGES" or (line.startswith("#") and in_nodes_section):
                break
            elif in_nodes_section and line and not line.startswith("#"):
                node = parse_node_line(line)
                if node:
                    nodes.append(node)
                    if node['is_viewpoint']:
                        viewpoint_nodes.append(node)
    
    except FileNotFoundError:
        print(f"错误: 文件不存在 {filepath}")
        return
    except Exception as e:
        print(f"错误: 读取文件失败 {filepath}: {e}")
        return
    
    print(f"总节点数: {len(nodes)}")
    print(f"视点节点数: {len(viewpoint_nodes)}")
    
    if not viewpoint_nodes:
        print("❌ 没有找到视点节点!")
        return
    
    # 分析视点收益信息
    print(f"\n🎯 视点收益分析:")
    print(f"{'节点ID':<8} {'位置':<20} {'观测得分':<12} {'集群距离':<12} {'可达性':<8}")
    print("-" * 70)
    
    obs_scores = []
    cluster_distances = []
    reachable_count = 0
    
    for vp in viewpoint_nodes:
        position_str = f"({vp['x']:.1f},{vp['y']:.1f},{vp['z']:.1f})"
        reachable_str = "是" if vp['is_reachable'] else "否"
        
        print(f"{vp['node_id']:<8} {position_str:<20} {vp['observation_score']:<12.1f} {vp['cluster_distance']:<12.2f} {reachable_str:<8}")
        
        if vp['observation_score'] > 0:
            obs_scores.append(vp['observation_score'])
        if vp['cluster_distance'] > 0:
            cluster_distances.append(vp['cluster_distance'])
        if vp['is_reachable']:
            reachable_count += 1
    
    # 统计信息
    print(f"\n📊 统计信息:")
    print(f"可达视点数量: {reachable_count}/{len(viewpoint_nodes)}")
    
    if obs_scores:
        print(f"观测得分 > 0 的视点: {len(obs_scores)}")
        print(f"观测得分范围: {min(obs_scores):.1f} ~ {max(obs_scores):.1f}")
        print(f"平均观测得分: {sum(obs_scores)/len(obs_scores):.1f}")
    else:
        print("⚠️  所有视点的观测得分都是 0!")
    
    if cluster_distances:
        print(f"集群距离 > 0 的视点: {len(cluster_distances)}")
        print(f"集群距离范围: {min(cluster_distances):.2f} ~ {max(cluster_distances):.2f}")
        print(f"平均集群距离: {sum(cluster_distances)/len(cluster_distances):.2f}")
    else:
        print("⚠️  所有视点的集群距离都 <= 0!")
    
    # 检查数据异常
    print(f"\n🔍 异常数据检查:")
    zero_obs_count = len([vp for vp in viewpoint_nodes if vp['observation_score'] == 0])
    negative_cluster_count = len([vp for vp in viewpoint_nodes if vp['cluster_distance'] < 0])
    
    print(f"观测得分为0的视点: {zero_obs_count}/{len(viewpoint_nodes)}")
    print(f"集群距离为负的视点: {negative_cluster_count}/{len(viewpoint_nodes)}")
    
    if zero_obs_count == len(viewpoint_nodes):
        print("❌ 所有视点的观测得分都是0，可能数据没有正确设置!")
    
    if negative_cluster_count == len(viewpoint_nodes):
        print("❌ 所有视点的集群距离都是负数，可能数据没有正确设置!")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python3 debug_viewpoint_benefits.py <文件路径或目录>")
        print("示例: python3 debug_viewpoint_benefits.py /path/to/topo_graph_file.txt")
        print("示例: python3 debug_viewpoint_benefits.py /path/to/collected_data/")
        return
    
    path = sys.argv[1]
    
    if os.path.isfile(path):
        # 分析单个文件
        analyze_topo_graph_file(path)
    elif os.path.isdir(path):
        # 分析目录中的所有topo_graph文件
        pattern = os.path.join(path, "**/topo_graph_*.txt")
        files = glob.glob(pattern, recursive=True)
        
        if not files:
            print(f"在目录 {path} 中没有找到 topo_graph_*.txt 文件")
            return
        
        # 按时间戳排序，分析最新的几个文件
        files.sort()
        recent_files = files[-3:]  # 最新的3个文件
        
        print(f"在目录 {path} 中找到 {len(files)} 个文件，分析最新的 {len(recent_files)} 个:")
        
        for filepath in recent_files:
            analyze_topo_graph_file(filepath)
    else:
        print(f"错误: 路径不存在 {path}")

if __name__ == "__main__":
    main()
