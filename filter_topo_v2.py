#!/usr/bin/env python3
"""
改进的拓扑图数据过滤脚本
功能：
1. 过滤掉 total_distance = 0 的无效数据
2. 过滤掉时间间隔太短的重复数据
3. 过滤掉距离间隔太短的重复数据
4. 检测探索终止点（viewpoints_visited = 0）并精确截取数据
"""

import os
import re
import sys
import shutil
from pathlib import Path

def extract_stats(content):
    """提取关键统计信息"""
    stats = {}
    
    # 提取total_distance
    distance_match = re.search(r'total_distance:\s+([\d.]+)', content)
    stats['total_distance'] = float(distance_match.group(1)) if distance_match else 0.0
    
    # 提取exploration_time
    time_match = re.search(r'exploration_time:\s+([\d.]+)', content)
    stats['exploration_time'] = float(time_match.group(1)) if time_match else 0.0
    
    # 提取viewpoints_visited
    viewpoints_match = re.search(r'viewpoints_visited:\s+(\d+)', content)
    stats['viewpoints_visited'] = int(viewpoints_match.group(1)) if viewpoints_match else -1
    
    # 提取文件时间戳（从第一行）
    timestamp_match = re.search(r'# Topo Graph Export.*?- ([\d.]+)', content)
    stats['timestamp'] = float(timestamp_match.group(1)) if timestamp_match else 0.0
    
    return stats

def should_keep_file(stats, prev_stats, min_distance=0.001, min_time_interval=0.5, min_distance_interval=0.1):
    """判断是否保留文件"""
    
    # 1. 如果是第一个文件（起始点已在外部处理），保留
    if prev_stats is None:
        return True, "首个有效文件"
    
    # 2. 过滤距离为0的文件（起始点除外，已在外部处理）
    if stats['total_distance'] <= min_distance:
        return False, "距离为零"
    
    # 3. 检查时间间隔
    time_diff = abs(stats['exploration_time'] - prev_stats['exploration_time'])
    if time_diff < min_time_interval:
        return False, f"时间间隔过短({time_diff:.3f}s < {min_time_interval}s)"
    
    # 4. 检查距离间隔
    distance_diff = abs(stats['total_distance'] - prev_stats['total_distance'])
    if distance_diff < min_distance_interval:
        return False, f"距离间隔过短({distance_diff:.3f}m < {min_distance_interval}m)"
    
    return True, "有效文件"

def filter_topo_files(input_dir, output_dir=None, min_time_interval=0.5, min_distance_interval=0.1, verbose=True):
    """过滤拓扑图文件"""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    # 设置输出目录
    if output_dir is None:
        output_path = input_path / "filtered_v2"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(exist_ok=True)
    
    # 获取所有txt文件并按文件名排序
    txt_files = sorted([f for f in input_path.glob("*.txt") if f.is_file()])
    
    if not txt_files:
        print(f"警告: 在 {input_dir} 中未找到txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件")
    print(f"最小时间间隔阈值: {min_time_interval}s")
    print(f"最小距离间隔阈值: {min_distance_interval}m")
    print(f"输出目录: {output_path}")
    print("-" * 60)
    
    # 第一步：找到最后一个 total_distance = 0.0 的文件作为起始点
    start_index = -1
    first_exploration_end_index = -1  # 第一个viewpoints_visited = 0的探索终止点
    all_stats = []
    
    # 预处理：读取所有文件的统计信息
    for i, txt_file in enumerate(txt_files):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            stats = extract_stats(content)
            stats['file'] = txt_file
            stats['index'] = i
            all_stats.append(stats)
            
            # 记录最后一个距离为0的文件
            if stats['total_distance'] == 0.0:
                start_index = i
            
            # 记录第一个viewpoints_visited = 0的探索终止点
            if first_exploration_end_index == -1 and stats['viewpoints_visited'] == 0 and stats['total_distance'] > 0.0:
                first_exploration_end_index = i
                
        except Exception as e:
            print(f"错误处理文件 {txt_file.name}: {e}")
    
    if start_index == -1:
        print("警告: 未找到 total_distance = 0.0 的起始文件，从第一个文件开始")
        start_index = 0
    else:
        print(f"找到探索起始点: {txt_files[start_index].name} (索引: {start_index})")
    
    if first_exploration_end_index != -1:
        print(f"找到探索终止点: {txt_files[first_exploration_end_index].name} (索引: {first_exploration_end_index})")
        print(f"探索终止点统计: viewpoints_visited={all_stats[first_exploration_end_index]['viewpoints_visited']}, total_distance={all_stats[first_exploration_end_index]['total_distance']:.3f}")
    
    # 第二步：从起始点开始进行过滤，到探索终止点为止
    end_index = first_exploration_end_index if first_exploration_end_index != -1 else len(all_stats) - 1
    total_files = end_index - start_index + 1
    kept_files = 0
    prev_stats = None
    filter_reasons = {}
    
    for i in range(start_index, end_index + 1):
        stats = all_stats[i]
        txt_file = stats['file']
        
        # 判断是否保留
        if i == start_index:
            # 起始文件必须保留
            should_keep, reason = True, "探索起始点"
        elif i == first_exploration_end_index:
            # 探索终止点必须保留
            should_keep, reason = True, "探索终止点"
        else:
            should_keep, reason = should_keep_file(stats, prev_stats, 
                                                 min_time_interval=min_time_interval,
                                                 min_distance_interval=min_distance_interval)
        
        if should_keep:
            # 复制文件
            output_file = output_path / txt_file.name
            shutil.copy2(txt_file, output_file)
            kept_files += 1
            prev_stats = stats
            
            if verbose:
                viewpoints_info = f"viewpoints:{stats['viewpoints_visited']:>3}" if stats['viewpoints_visited'] != -1 else "viewpoints: N/A"
                print(f"✓ {txt_file.name:<40} | 时间:{stats['exploration_time']:>8.3f}s | 距离:{stats['total_distance']:>8.3f} | {viewpoints_info} | {reason}")
        else:
            # 记录过滤原因
            filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
            
            if verbose:
                viewpoints_info = f"viewpoints:{stats['viewpoints_visited']:>3}" if stats['viewpoints_visited'] != -1 else "viewpoints: N/A"
                print(f"✗ {txt_file.name:<40} | 时间:{stats['exploration_time']:>8.3f}s | 距离:{stats['total_distance']:>8.3f} | {viewpoints_info} | {reason}")
    
    # 统计被过滤掉的探索终止后文件数量
    post_exploration_files = len(all_stats) - end_index - 1
    if post_exploration_files > 0:
        filter_reasons["探索终止后文件"] = post_exploration_files
        if verbose:
            print(f"\n过滤掉探索终止后的 {post_exploration_files} 个文件")
    
    # 输出统计结果
    print("-" * 60)
    print(f"处理完成!")
    print(f"扫描文件数: {len(all_stats)}")
    print(f"有效范围文件数: {total_files} (从起始点到终止点)")
    print(f"保留文件: {kept_files}")
    print(f"过滤文件: {len(all_stats) - kept_files}")
    print(f"保留率: {kept_files/len(all_stats)*100:.1f}%")
    
    print("\n过滤原因统计:")
    for reason, count in filter_reasons.items():
        print(f"  {reason}: {count}个文件")
    
    return kept_files

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='过滤拓扑图数据，移除无效和重复数据')
    parser.add_argument('input_dir', help='输入目录路径')
    parser.add_argument('-o', '--output', help='输出目录路径 (默认: input_dir/filtered_v2)')
    parser.add_argument('-t', '--time-interval', type=float, default=0.5, 
                       help='最小时间间隔阈值(秒) (默认: 0.5)')
    parser.add_argument('-d', '--distance-interval', type=float, default=0.5,
                       help='最小距离间隔阈值(米) (默认: 0.5)')
    parser.add_argument('-q', '--quiet', action='store_true', help='静默模式')
    
    args = parser.parse_args()
    
    filter_topo_files(
        input_dir=args.input_dir,
        output_dir=args.output,
        min_time_interval=args.time_interval,
        min_distance_interval=args.distance_interval,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main()
