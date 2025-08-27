#!/usr/bin/env python3
"""
拓扑图数据过滤脚本
过滤掉探索前后的无效数据（total_distance: 0.000000）
"""

import os
import sys
import argparse
import glob
from pathlib import Path

def extract_distance(file_path):
    """从文件中提取total_distance值"""
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('total_distance:'):
                    distance = float(line.split(':')[1].strip())
                    return distance
        return 0.0
    except:
        return 0.0

def filter_episode_data(input_dir, output_dir=None, distance_threshold=0.001):
    """
    过滤episode数据
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，如果为None则在输入目录创建filtered子目录
        distance_threshold: 距离阈值，小于此值的文件被认为是无效的
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    # 设置输出目录
    if output_dir is None:
        output_path = input_path / "filtered"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(exist_ok=True)
    
    # 获取所有txt文件并排序
    txt_files = sorted(glob.glob(str(input_path / "*.txt")))
    
    if not txt_files:
        print(f"警告: 在 {input_dir} 中未找到txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个文件")
    
    # 分析文件，找到有效的探索数据段
    valid_files = []
    for file_path in txt_files:
        distance = extract_distance(file_path)
        if distance >= distance_threshold:
            valid_files.append(file_path)
    
    print(f"有效文件数量: {len(valid_files)} / {len(txt_files)}")
    
    if not valid_files:
        print("警告: 未找到有效的探索数据")
        return
    
    # 复制有效文件到输出目录
    copied_count = 0
    for src_file in valid_files:
        src_path = Path(src_file)
        dst_path = output_path / src_path.name
        
        try:
            # 简单的文件复制
            with open(src_path, 'r') as src, open(dst_path, 'w') as dst:
                dst.write(src.read())
            copied_count += 1
        except Exception as e:
            print(f"复制文件失败 {src_path.name}: {e}")
    
    # 统计信息
    first_file = Path(valid_files[0]).stem
    last_file = Path(valid_files[-1]).stem
    
    print(f"\\n过滤完成:")
    print(f"  输入目录: {input_path}")
    print(f"  输出目录: {output_path}")
    print(f"  复制文件: {copied_count}")
    print(f"  时间范围: {first_file} -> {last_file}")
    print(f"  过滤比例: {copied_count/len(txt_files)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(
        description="过滤拓扑图数据，移除探索前后的无效数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 过滤当前目录下的数据，结果存放在filtered子目录
  python filter_topo_data.py ./topo_outputs
  
  # 过滤数据并指定输出目录
  python filter_topo_data.py ./topo_outputs -o ./clean_data
  
  # 使用更严格的距离阈值
  python filter_topo_data.py ./topo_outputs -t 0.01
        """)
    
    parser.add_argument('input_dir', 
                       help='包含拓扑图文件的输入目录')
    parser.add_argument('-o', '--output', 
                       help='输出目录 (默认: 输入目录/filtered)')
    parser.add_argument('-t', '--threshold', type=float, default=0.001,
                       help='距离阈值，小于此值视为无效数据 (默认: 0.001)')
    
    args = parser.parse_args()
    
    filter_episode_data(args.input_dir, args.output, args.threshold)

if __name__ == "__main__":
    main()
