#!/usr/bin/env python3
"""
深度分析数据集中的异常样本
检查是否存在非episode终止态的无效数据
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import argparse

def deep_analyze_batch_file(batch_file_path):
    """深度分析单个批次文件中的异常样本"""
    anomalies = []
    
    try:
        with h5py.File(batch_file_path, 'r') as f:
            # 获取所有数据
            viewpoints = f['viewpoints'][:]  # [N, 25, 1]
            viewpoint_padding_mask = f['viewpoint_padding_mask'][:]  # [N, 1, 25]
            node_inputs = f['node_inputs'][:]  # [N, 2500, 9]
            node_padding_mask = f['node_padding_mask'][:]  # [N, 1, 2500]
            rewards = f['rewards'][:]
            actions = f['actions'][:]
            dones = f['dones'][:]
            current_index = f['current_index'][:]
            
            # 额外数据（如果存在）
            if 'episode_id' in f:
                episode_ids = f['episode_id'][:]
            else:
                episode_ids = np.zeros(len(viewpoints))
            
            if 'timestep' in f:
                timesteps = f['timestep'][:]
            else:
                timesteps = np.arange(len(viewpoints))
            
            N = len(viewpoints)
            
            for i in range(N):
                # 计算有效视点数和有效节点数
                vp_mask = viewpoint_padding_mask[i, 0, :]  # [25]
                node_mask = node_padding_mask[i, 0, :]     # [2500]
                
                valid_vps = np.sum(~vp_mask)  # False表示有效
                valid_nodes = np.sum(~node_mask)
                
                # 分析异常情况
                anomaly_info = {
                    'sample_idx': i,
                    'batch_file': os.path.basename(batch_file_path),
                    'episode_id': episode_ids[i] if len(episode_ids) > i else -1,
                    'timestep': timesteps[i] if len(timesteps) > i else -1,
                    'action': actions[i],
                    'reward': rewards[i],
                    'done': dones[i],
                    'current_index': current_index[i, 0, 0] if current_index.shape[-1] > 0 else -1,
                    'valid_vps': valid_vps,
                    'valid_nodes': valid_nodes,
                    'anomalies': []
                }
                
                # 检查各种异常情况
                
                # 1. valid_vps=0 但不是done状态
                if valid_vps == 0 and not dones[i]:
                    anomaly_info['anomalies'].append("valid_vps=0_but_not_done")
                
                # 2. valid_vps=0 但在episode中间位置
                if valid_vps == 0 and i < N - 1:  # 不是最后一个样本
                    # 检查后续样本是否还有数据
                    if i < N - 5:  # 还有5个以上后续样本
                        anomaly_info['anomalies'].append("valid_vps=0_in_middle")
                
                # 3. 有效节点数为0
                if valid_nodes == 0:
                    anomaly_info['anomalies'].append("valid_nodes=0")
                
                # 4. current_index超出有效节点范围
                if anomaly_info['current_index'] >= valid_nodes and valid_nodes > 0:
                    anomaly_info['anomalies'].append("current_index_out_of_range")
                
                # 5. action超出有效视点范围
                if actions[i] >= valid_vps and valid_vps > 0:
                    anomaly_info['anomalies'].append("action_out_of_range")
                
                # 6. 节点特征异常
                node_features = node_inputs[i]
                valid_node_features = node_features[~node_mask]
                if len(valid_node_features) > 0:
                    # 检查是否有NaN或Inf
                    if np.isnan(valid_node_features).any():
                        anomaly_info['anomalies'].append("node_features_nan")
                    if np.isinf(valid_node_features).any():
                        anomaly_info['anomalies'].append("node_features_inf")
                    
                    # 检查特征值范围是否合理
                    feature_max = np.max(np.abs(valid_node_features))
                    if feature_max > 1000:  # 特征值过大
                        anomaly_info['anomalies'].append(f"node_features_too_large_{feature_max:.1f}")
                
                # 7. 视点数据异常
                valid_viewpoint_data = viewpoints[i][~vp_mask]
                if len(valid_viewpoint_data) > 0:
                    vp_max = np.max(valid_viewpoint_data)
                    vp_min = np.min(valid_viewpoint_data)
                    
                    # 视点索引应该在有效节点范围内
                    if vp_max >= valid_nodes:
                        anomaly_info['anomalies'].append("viewpoint_index_out_of_node_range")
                    if vp_min < 0:
                        anomaly_info['anomalies'].append("viewpoint_index_negative")
                
                # 8. 奖励异常
                if np.isnan(rewards[i]) or np.isinf(rewards[i]):
                    anomaly_info['anomalies'].append("reward_nan_inf")
                if abs(rewards[i]) > 10:  # 奖励过大
                    anomaly_info['anomalies'].append(f"reward_too_large_{rewards[i]:.3f}")
                
                # 只记录有异常的样本
                if anomaly_info['anomalies']:
                    anomalies.append(anomaly_info)
    
    except Exception as e:
        print(f"❌ 处理文件 {batch_file_path} 时出错: {e}")
        return []
    
    return anomalies

def analyze_dataset_anomalies(data_path):
    """分析整个数据集的异常情况"""
    print(f"🔍 深度分析数据集异常: {data_path}")
    
    # 获取所有批次文件
    batch_files = []
    for f in os.listdir(data_path):
        if f.endswith('.h5') and ('batch' in f or f.startswith('dataset_batch_')):
            batch_files.append(os.path.join(data_path, f))
    
    batch_files.sort()
    print(f"📁 找到 {len(batch_files)} 个批次文件")
    
    all_anomalies = []
    anomaly_stats = Counter()
    
    # 分析每个文件
    for i, batch_file in enumerate(batch_files):
        if i % 20 == 0:  # 只显示进度
            print(f"📊 处理文件 {i+1}/{len(batch_files)}")
        
        anomalies = deep_analyze_batch_file(batch_file)
        all_anomalies.extend(anomalies)
        
        # 统计异常类型
        for anomaly in anomalies:
            for anomaly_type in anomaly['anomalies']:
                anomaly_stats[anomaly_type] += 1
    
    return all_anomalies, anomaly_stats

def analyze_episode_structure(anomalies):
    """分析异常样本的episode结构"""
    print(f"\n🔬 分析异常样本的episode结构:")
    
    # 按batch文件分组
    file_anomalies = defaultdict(list)
    for anomaly in anomalies:
        file_anomalies[anomaly['batch_file']].append(anomaly)
    
    problematic_files = []
    
    for batch_file, file_anomalies_list in file_anomalies.items():
        # 按sample_idx排序
        file_anomalies_list.sort(key=lambda x: x['sample_idx'])
        
        print(f"\n📄 文件: {batch_file}")
        print(f"   异常样本数: {len(file_anomalies_list)}")
        
        # 检查异常样本的分布
        sample_indices = [a['sample_idx'] for a in file_anomalies_list]
        if sample_indices:
            print(f"   异常样本位置: {min(sample_indices)} - {max(sample_indices)}")
            
            # 检查是否在episode末尾
            max_idx = max(sample_indices)
            
            # 尝试获取文件总样本数
            try:
                with h5py.File(os.path.join('/home/amax/EPIC/datasets', batch_file), 'r') as f:
                    total_samples = len(f['actions'])
                
                end_ratio = max_idx / total_samples
                print(f"   最大异常位置比例: {end_ratio:.1%} ({max_idx}/{total_samples})")
                
                if end_ratio < 0.9:  # 异常出现在前90%
                    problematic_files.append({
                        'file': batch_file,
                        'anomalies': file_anomalies_list,
                        'end_ratio': end_ratio,
                        'total_samples': total_samples
                    })
                    print(f"   ⚠️  异常出现在episode中间，需要深入检查")
            except:
                print(f"   ❌ 无法读取文件总样本数")
        
        # 显示前几个异常样本的详情
        for j, anomaly in enumerate(file_anomalies_list[:3]):
            anomaly_types_str = ", ".join(anomaly['anomalies'])
            print(f"     样本 {anomaly['sample_idx']}: {anomaly_types_str}")
            print(f"       valid_vps={anomaly['valid_vps']}, done={anomaly['done']}, "
                  f"action={anomaly['action']}, reward={anomaly['reward']:.4f}")
    
    return problematic_files

def detailed_file_analysis(file_path):
    """对问题文件进行详细分析"""
    print(f"\n🔍 详细分析问题文件: {os.path.basename(file_path)}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            viewpoint_padding_mask = f['viewpoint_padding_mask'][:]
            dones = f['dones'][:]
            rewards = f['rewards'][:]
            actions = f['actions'][:]
            
            N = len(dones)
            
            # 计算每个样本的有效视点数
            valid_vps_sequence = []
            for i in range(N):
                vp_mask = viewpoint_padding_mask[i, 0, :]
                valid_vps = np.sum(~vp_mask)
                valid_vps_sequence.append(valid_vps)
            
            # 找到所有valid_vps=0的位置
            zero_vps_positions = [i for i, vps in enumerate(valid_vps_sequence) if vps == 0]
            done_positions = [i for i, done in enumerate(dones) if done]
            
            print(f"   总样本数: {N}")
            print(f"   valid_vps=0的位置: {zero_vps_positions}")
            print(f"   done=True的位置: {done_positions}")
            
            # 检查连续的valid_vps变化
            print(f"\n   有效视点数变化序列（显示前20个和后20个）:")
            for i in range(min(20, N)):
                status = ""
                if i in zero_vps_positions:
                    status += " [ZERO_VPS]"
                if i in done_positions:
                    status += " [DONE]"
                print(f"     样本 {i:3d}: valid_vps={valid_vps_sequence[i]:2d}, done={dones[i]}, "
                      f"action={actions[i]:2d}, reward={rewards[i]:6.3f}{status}")
            
            if N > 40:
                print("     ...")
                for i in range(max(20, N-20), N):
                    status = ""
                    if i in zero_vps_positions:
                        status += " [ZERO_VPS]"
                    if i in done_positions:
                        status += " [DONE]"
                    print(f"     样本 {i:3d}: valid_vps={valid_vps_sequence[i]:2d}, done={dones[i]}, "
                          f"action={actions[i]:2d}, reward={rewards[i]:6.3f}{status}")
            
            # 检查是否有非终止位置的zero_vps
            non_terminal_zero_vps = []
            for pos in zero_vps_positions:
                if pos < N - 1 and not dones[pos]:  # 不是最后一个且不是done状态
                    non_terminal_zero_vps.append(pos)
            
            if non_terminal_zero_vps:
                print(f"\n   ⚠️  发现 {len(non_terminal_zero_vps)} 个非终止位置的zero_vps:")
                for pos in non_terminal_zero_vps:
                    print(f"     位置 {pos}: valid_vps=0, done={dones[pos]}, 后续还有{N-pos-1}个样本")
    
    except Exception as e:
        print(f"❌ 详细分析失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='深度分析数据集中的异常样本')
    parser.add_argument('--data_path', default='/home/amax/EPIC/datasets_v3', 
                       help='数据集目录路径')
    parser.add_argument('--detailed_analysis', action='store_true',
                       help='对问题文件进行详细分析')
    
    args = parser.parse_args()
    
    print("🔍 EPIC 3D数据集异常深度分析")
    print("=" * 60)
    
    # 分析异常
    all_anomalies, anomaly_stats = analyze_dataset_anomalies(args.data_path)
    
    print(f"\n📊 异常统计:")
    print(f"  总异常样本数: {len(all_anomalies)}")
    
    if not all_anomalies:
        print("✅ 未发现任何异常！数据集质量良好。")
        return
    
    print(f"\n📋 异常类型统计:")
    for anomaly_type, count in anomaly_stats.most_common():
        print(f"  {anomaly_type}: {count} 次")
    
    # 分析episode结构
    problematic_files = analyze_episode_structure(all_anomalies)
    
    if problematic_files:
        print(f"\n⚠️  发现 {len(problematic_files)} 个可能有问题的文件:")
        for pf in problematic_files:
            print(f"  {pf['file']}: 异常位置比例 {pf['end_ratio']:.1%}")
        
        # 详细分析第一个问题文件
        if args.detailed_analysis and problematic_files:
            for i in range(len(problematic_files)):
                first_problem_file = os.path.join(args.data_path, problematic_files[i]['file'])
                detailed_file_analysis(first_problem_file)
    else:
        print("✅ 所有异常都出现在episode末尾，数据结构正常。")
    
    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()
