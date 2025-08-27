"""
EPIC 3D数据处理模块

将EPIC 3D探索系统收集的topo_graph数据转换为适用于离线强化学习训练的格式。
无需修改现有的C++数据采集Pipeline，完全基于现有数据文件。
"""

import os
import glob
import numpy as np
import torch
import h5py
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from collections import defaultdict, namedtuple
import re
import time

# 定义EPIC3D数据结构
EPIC3DTimeStep = namedtuple('EPIC3DTimeStep', [
    'timestamp',          # 时间戳
    'exploration_stats',  # 探索统计信息字典
    'nodes',             # 节点信息列表
    'edges',             # 边信息列表
    'viewpoints',        # 视点节点信息
    'current_pos',       # 当前位置 (x,y,z)
    'trajectory_info'    # 轨迹信息字典
])

class EPIC3DDataParser:
    """解析单个topo_graph文件"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_topo_file(self, file_path: str) -> EPIC3DTimeStep:
        """
        解析单个topo_graph文件
        
        Args:
            file_path: topo_graph文件路径
            
        Returns:
            EPIC3DTimeStep: 解析后的时间步数据
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}")
        
        # 解析时间戳
        timestamp = self._extract_timestamp(lines[0])
        
        # 解析探索统计
        exploration_stats = self._parse_exploration_stats(lines)
        
        # 解析节点和边数据
        nodes, edges = self._parse_graph_data(lines)
        
        # 提取视点信息
        viewpoints = self._extract_viewpoints(nodes)
        
        # 找到当前位置
        current_pos = self._find_current_position(nodes)
        
        # 构建轨迹信息
        trajectory_info = self._build_trajectory_info(nodes)
        
        return EPIC3DTimeStep(
            timestamp=timestamp,
            exploration_stats=exploration_stats,
            nodes=nodes,
            edges=edges,
            viewpoints=viewpoints,
            current_pos=current_pos,
            trajectory_info=trajectory_info
        )
    
    def _extract_timestamp(self, header_line: str) -> float:
        """从文件头提取时间戳"""
        match = re.search(r'(\d+\.\d+)$', header_line.strip())
        if match:
            return float(match.group(1))
        else:
            # 尝试从文件名提取时间戳
            self.logger.warning(f"Could not extract timestamp from header: {header_line}")
            return 0.0
    
    def _parse_exploration_stats(self, lines: List[str]) -> Dict:
        """解析探索统计信息"""
        stats = {}
        in_stats_section = False
        
        for line in lines:
            line = line.strip()
            if line == "# ===== EXPLORATION STATISTICS =====":
                in_stats_section = True
                continue
            elif line.startswith("# 节点格式:") or line == "NODES":
                break
            elif in_stats_section:
                # 跳过注释行（以#开头的行）
                if line.startswith("#"):
                    continue
                # 解析统计数据行（key: value格式）
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        # 尝试转换为数值
                        if '.' in value:
                            stats[key] = float(value)
                        else:
                            stats[key] = int(value)
                    except ValueError:
                        stats[key] = value
        
        return stats
    
    def _parse_graph_data(self, lines: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """解析节点和边数据"""
        nodes = []
        edges = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line == "NODES":
                current_section = "nodes"
                continue
            elif line == "EDGES":
                current_section = "edges"
                continue
            elif line.startswith("#") or not line:
                continue
            
            if current_section == "nodes":
                try:
                    node = self._parse_node_line(line)
                    nodes.append(node)
                except Exception as e:
                    self.logger.warning(f"Failed to parse node line: {line}, error: {e}")
                    
            elif current_section == "edges":
                try:
                    edge = self._parse_edge_line(line)
                    edges.append(edge)
                except Exception as e:
                    self.logger.warning(f"Failed to parse edge line: {line}, error: {e}")
        
        return nodes, edges
    
    def _parse_node_line(self, line: str) -> Dict:
        """解析节点数据行"""
        parts = line.split()
        
        if len(parts) < 14:
            raise ValueError(f"Invalid node line format, expected 14 fields, got {len(parts)}: {line}")
        
        return {
            'node_id': int(parts[0]),
            'position': np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32),
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
    
    def _parse_edge_line(self, line: str) -> Dict:
        """解析边数据行"""
        parts = line.split()
        
        if len(parts) < 5:
            raise ValueError(f"Invalid edge line format, expected 5 fields, got {len(parts)}: {line}")
        
        return {
            'edge_id': int(parts[0]),
            'from_node_id': int(parts[1]),
            'to_node_id': int(parts[2]),
            'weight': float(parts[3]),
            'is_reachable': bool(int(parts[4]))
        }
    
    def _extract_viewpoints(self, nodes: List[Dict]) -> List[Dict]:
        """提取视点节点"""
        return [node for node in nodes if node['is_viewpoint']]
    
    def _find_current_position(self, nodes: List[Dict]) -> Optional[np.ndarray]:
        """找到当前位置"""
        current_nodes = [node for node in nodes if node['is_current']]
        if current_nodes:
            return current_nodes[0]['position'].copy()
        else:
            # 如果没有明确的当前节点，尝试找到tsp_order_index为0的节点
            tsp_zero_nodes = [node for node in nodes if node['tsp_order_index'] == 0]
            if tsp_zero_nodes:
                return tsp_zero_nodes[0]['position'].copy()
            else:
                self.logger.warning("Could not find current position")
                return None
    
    def _build_trajectory_info(self, nodes: List[Dict]) -> Dict:
        """构建轨迹信息"""
        history_nodes = [node for node in nodes if node['is_history']]
        viewpoint_nodes = [node for node in nodes if node['is_viewpoint']]
        reachable_nodes = [node for node in nodes if node['is_reachable']]
        
        return {
            'history_positions': [node['position'].copy() for node in history_nodes],
            'total_nodes': len(nodes),
            'viewpoint_nodes': len(viewpoint_nodes),
            'reachable_nodes': len(reachable_nodes),
            'max_tsp_order': max([node['tsp_order_index'] for node in nodes if node['tsp_order_index'] >= 0], default=-1)
        }


class EPIC3DEpisodeProcessor:
    """处理单个episode的所有时间步数据"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.parser = EPIC3DDataParser()
        self.logger = logging.getLogger(__name__)
    
    def process_episode_directory(self, episode_dir: str) -> Dict:
        """
        处理单个episode目录
        
        Args:
            episode_dir: episode目录路径
            
        Returns:
            Dict: 处理后的episode数据，如果episode不完整则返回None
        """
        filtered_data_dir = os.path.join(episode_dir, 'filtered_data')
        
        if not os.path.exists(filtered_data_dir):
            raise ValueError(f"Filtered data directory not found: {filtered_data_dir}")
        
        # 获取所有topo_graph文件并按时间戳排序
        topo_files = sorted(glob.glob(os.path.join(filtered_data_dir, 'topo_graph_*.txt')),
                           key=lambda x: self._extract_timestamp_from_filename(x))
        
        if not topo_files:
            raise ValueError(f"No topo_graph files found in {filtered_data_dir}")
        
        self.logger.info(f"Processing {len(topo_files)} time steps from {episode_dir}")
        
        # 解析所有时间步
        time_steps = []
        for i, file_path in enumerate(topo_files):
            try:
                time_step = self.parser.parse_topo_file(file_path)
                time_steps.append(time_step)
            except Exception as e:
                self.logger.warning(f"Failed to parse {file_path}: {e}")
                continue
        
        if not time_steps:
            raise ValueError(f"No valid time steps found in {episode_dir}")
        
        # 检查episode完整性 - 最后一帧应该是done状态
        if not self._validate_episode_completion(time_steps, episode_dir):
            self.logger.warning(f"Episode incomplete, discarding: {episode_dir}")
            return None
        
        # 构建episode数据
        episode_data = self._build_episode_data(time_steps, episode_dir)
        
        return episode_data
    
    def _extract_timestamp_from_filename(self, filename: str) -> float:
        """从文件名提取时间戳用于排序"""
        basename = os.path.basename(filename)
        match = re.search(r'topo_graph_(\d+\.\d+)\.txt', basename)
        return float(match.group(1)) if match else 0.0
    
    def _validate_episode_completion(self, time_steps: List[EPIC3DTimeStep], episode_dir: str) -> bool:
        """
        验证episode是否完整完成探索
        
        规则：最后一帧的viewpoints_visited应该为0，表示探索结束后的done状态
        如果不为0，说明探索过程被意外中断，该episode应该被舍弃
        
        Args:
            time_steps: 时间步序列
            episode_dir: episode目录路径（用于日志）
            
        Returns:
            bool: True表示episode完整，False表示应该舍弃
        """
        if not time_steps:
            return False
        
        last_step = time_steps[-1]
        viewpoints_visited = last_step.exploration_stats.get('viewpoints_visited', -1)
        
        # 检查最后一帧的viewpoints_visited
        if viewpoints_visited == 0:
            self.logger.info(f"✅ Episode完整: {os.path.basename(episode_dir)} "
                           f"(最后一帧viewpoints_visited={viewpoints_visited})")
            return True
        else:
            self.logger.warning(f"❌ Episode不完整: {os.path.basename(episode_dir)} "
                              f"(最后一帧viewpoints_visited={viewpoints_visited}，应为0)")
            
            # 额外检查：记录一些统计信息帮助调试
            total_steps = len(time_steps)
            first_step_viewpoints = time_steps[0].exploration_stats.get('viewpoints_visited', -1)
            exploration_time = last_step.exploration_stats.get('exploration_time', 0)
            
            self.logger.warning(f"  - 总时间步数: {total_steps}")
            self.logger.warning(f"  - 首帧viewpoints_visited: {first_step_viewpoints}")
            self.logger.warning(f"  - 探索时长: {exploration_time:.1f}s")
            self.logger.warning(f"  - 该episode将被舍弃")
            
            return False
    
    def _build_episode_data(self, time_steps: List[EPIC3DTimeStep], episode_dir: str) -> Dict:
        """
        将时间步数据转换为episode格式
        
        关键修正：n+1个观测对应n个transition
        - time_steps包含n+1个观测(状态)
        - 生成n个{s_t, a_t, r_{t+1}}元组用于RL训练
        """
        total_observations = len(time_steps)
        transition_count = total_observations - 1  # n个transition
        
        # 提取episode元数据
        episode_metadata = self._extract_episode_metadata(time_steps, episode_dir)
        
        # 构建状态序列 (包含s_0到s_n，共n+1个状态)
        states = self._build_state_sequence(time_steps)
        
        # 构建动作序列 (只有n个动作: a_0到a_{n-1})
        actions = self._build_action_sequence(time_steps[:transition_count])
        
        # 构建奖励序列 (只有n个奖励: r_1到r_n) 
        rewards = self._build_reward_sequence(time_steps)
        
        # 构建done标志 (只有n个done标志)
        dones = self._build_done_sequence(time_steps[:transition_count])
        
        episode_data = {
            'metadata': episode_metadata,
            'states': states,  # 保持完整的n+1个状态
            'actions': actions,  # n个动作
            'rewards': rewards,  # n个奖励
            'dones': dones,  # n个done标志
            'episode_length': transition_count,  # episode长度应该是transition数量
            'total_observations': total_observations  # 记录总观测数
        }
        
        return episode_data
    
    def _extract_episode_metadata(self, time_steps: List[EPIC3DTimeStep], episode_dir: str) -> Dict:
        """提取episode元数据"""
        first_step = time_steps[0]
        last_step = time_steps[-1]
        
        # 读取配置文件
        config_file = os.path.join(episode_dir, 'config.txt')
        episode_config = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            episode_config[key.strip()] = value.strip()
            except Exception as e:
                self.logger.warning(f"Failed to read config file {config_file}: {e}")
        
        return {
            'episode_name': os.path.basename(episode_dir),
            'start_time': first_step.timestamp,
            'end_time': last_step.timestamp,
            'duration': last_step.timestamp - first_step.timestamp,
            'final_exploration_stats': last_step.exploration_stats,
            'config': episode_config,
            'num_time_steps': len(time_steps)
        }
    
    def _build_state_sequence(self, time_steps: List[EPIC3DTimeStep]) -> Dict:
        """构建状态序列"""
        max_nodes = self.config.get('max_nodes', 500)
        max_viewpoints = self.config.get('max_viewpoints', 100)
        k_size = self.config.get('k_size', 20)
        node_feature_dim = self.config.get('node_feature_dim', 9)
        
        # 标准化参数
        position_scale = self.config.get('position_scale', 100.0)
        observation_score_scale = self.config.get('observation_score_scale', 50.0)
        cluster_distance_scale = self.config.get('cluster_distance_scale', 20.0)
        distance_scale = self.config.get('distance_scale', 50.0)
        
        episode_length = len(time_steps)
        
        # 初始化状态张量
        node_inputs = np.zeros((episode_length, max_nodes, node_feature_dim), dtype=np.float32)
        node_padding_mask = np.ones((episode_length, 1, max_nodes), dtype=bool)  # True = padding
        current_index = np.zeros((episode_length, 1, 1), dtype=np.int64)
        viewpoint_indices = np.zeros((episode_length, max_viewpoints, 1), dtype=np.float32)
        viewpoint_padding_mask = np.ones((episode_length, 1, max_viewpoints), dtype=bool)
        adj_list = np.full((episode_length, max_nodes, k_size), -1, dtype=np.int64)
        
        for t, time_step in enumerate(time_steps):
            # 处理节点特征
            nodes = time_step.nodes
            num_nodes = min(len(nodes), max_nodes)
            
            if num_nodes == 0:
                self.logger.warning(f"Time step {t} has no nodes")
                continue
            
            current_pos = time_step.current_pos
            if current_pos is None:
                current_pos = np.array([0, 0, 0], dtype=np.float32)
                
            # 构建节点特征矩阵
            current_node_idx = 0
            for i, node in enumerate(nodes[:num_nodes]):
                # 计算相对位置并标准化
                rel_pos = (node['position'] - current_pos) / position_scale
                
                # 标准化其他特征
                obs_score = max(0, node['observation_score']) / observation_score_scale
                cluster_dist = max(0, node['cluster_distance']) / cluster_distance_scale if node['cluster_distance'] > 0 else 0
                distance = max(0, node['distance']) / distance_scale if node['distance'] > 0 else 0
                
                # 构建特征向量: [rel_x, rel_y, rel_z, obs_score, cluster_dist, is_vp, visited, dist, centrality]
                feature_vector = np.array([
                    rel_pos[0], rel_pos[1], rel_pos[2],      # 相对3D位置
                    obs_score,                                # 标准化观测得分
                    cluster_dist,                            # 标准化路径代价
                    float(node['is_viewpoint']),            # 节点类型
                    float(node['is_history']),              # 访问状态
                    distance,                               # 距离特征
                    0.5                                     # 占位的中心性得分
                ], dtype=np.float32)
                
                node_inputs[t, i] = feature_vector
                
                # 找到当前节点索引
                if node['is_current']:
                    current_node_idx = i
            
            current_index[t, 0, 0] = current_node_idx
            
            # 设置节点padding mask
            if num_nodes > 0:
                node_padding_mask[t, 0, :num_nodes] = False
            
            # 构建邻接表
            self._build_adjacency_list(time_step.edges, adj_list[t], num_nodes, k_size)
            
            # 提取视点索引
            viewpoint_nodes = [i for i, node in enumerate(nodes[:num_nodes]) if node['is_viewpoint']]
            num_viewpoints = min(len(viewpoint_nodes), max_viewpoints)
            
            for i, vp_idx in enumerate(viewpoint_nodes[:num_viewpoints]):
                viewpoint_indices[t, i, 0] = vp_idx
            
            # 设置视点padding mask
            if num_viewpoints > 0:
                viewpoint_padding_mask[t, 0, :num_viewpoints] = False
        
        return {
            'node_inputs': node_inputs,
            'node_padding_mask': node_padding_mask,
            'current_index': current_index,
            'viewpoints': viewpoint_indices,
            'viewpoint_padding_mask': viewpoint_padding_mask,
            'adj_list': adj_list
        }
    
    def _build_adjacency_list(self, edges: List[Dict], adj_list: np.ndarray, num_nodes: int, k_size: int):
        """构建邻接表"""
        # 构建邻接字典
        adjacency_dict = defaultdict(list)
        
        for edge in edges:
            from_id = edge['from_node_id']
            to_id = edge['to_node_id']
            
            # 确保节点ID在有效范围内
            if 0 <= from_id < num_nodes and 0 <= to_id < num_nodes:
                adjacency_dict[from_id].append(to_id)
        
        # 填充邻接表
        for node_id in range(num_nodes):
            neighbors = adjacency_dict.get(node_id, [])
            num_neighbors = min(len(neighbors), k_size)
            if num_neighbors > 0:
                adj_list[node_id, :num_neighbors] = neighbors[:num_neighbors]
    
    def _build_action_sequence(self, time_steps: List[EPIC3DTimeStep]) -> np.ndarray:
        """
        构建动作序列 - 基于TSP顺序信息
        
        注意：只为前n个状态生成动作，最后一个状态不产生动作
        """
        transition_count = len(time_steps)  # 输入已经是前n个time_steps
        actions = np.zeros(transition_count, dtype=np.int64)
        
        for t in range(transition_count):
            current_step = time_steps[t]
            
            # 通过TSP顺序推断动作
            action_idx = self._infer_action_from_tsp_order(current_step)
            actions[t] = action_idx
        
        return actions
    
    def _infer_action_from_tsp_order(self, time_step: EPIC3DTimeStep) -> int:
        """通过TSP顺序推断动作索引"""
        viewpoints = time_step.viewpoints
        if not viewpoints:
            return 0
        
        # 查找tsp_order_index = 1的视点作为下一个目标(动作)
        for i, vp in enumerate(viewpoints):
            if vp.get('tsp_order_index') == 1:
                return i
        
        # 如果没有找到tsp_order_index = 1的视点，返回第一个可达视点
        reachable_viewpoints = [i for i, vp in enumerate(viewpoints) if vp.get('is_reachable', False)]
        return reachable_viewpoints[0] if reachable_viewpoints else 0
    
    def _build_reward_sequence(self, time_steps: List[EPIC3DTimeStep]) -> np.ndarray:
        """
        构建奖励序列 - 基于距离和探索面积的变化
        
        注意：奖励序列长度=transition数量=总观测数-1
        reward[i]对应从state[i]到state[i+1]的transition奖励
        """
        total_observations = len(time_steps)
        transition_count = total_observations - 1
        rewards = np.zeros(transition_count, dtype=np.float32)
        
        for t in range(transition_count):
            # t对应从time_steps[t]到time_steps[t+1]的transition
            prev_stats = time_steps[t].exploration_stats
            curr_stats = time_steps[t+1].exploration_stats
            
            # 获取统计数据 - 使用实际的字段名
            prev_distance = prev_stats.get('total_distance', 0.0)
            curr_distance = curr_stats.get('total_distance', 0.0)
            prev_area = prev_stats.get('exploration_area', 0.0)  # 修正字段名
            curr_area = curr_stats.get('exploration_area', 0.0)  # 修正字段名
            
            # 计算变化量
            distance_change = curr_distance - prev_distance
            
            # 奖励计算：探索新区域给正奖励，距离增加给负奖励
            distance_penalty = -distance_change * 0.05  # 移动距离的惩罚
            
            # 组合奖励
            total_reward = distance_penalty
            rewards[t] = total_reward
            
            # 调试信息：打印前几步的奖励计算
            if t <= 5:
                self.logger.debug(f"Transition {t}: from step {t} to step {t+1}, "
                                f"distance_change={distance_change:.3f}, "
                                f"distance_penalty={distance_penalty:.3f}, "
                                f"total_reward={total_reward:.3f}")
        
        return rewards
    
    def _build_done_sequence(self, time_steps: List[EPIC3DTimeStep]) -> np.ndarray:
        """
        构建episode结束标志
        
        注意：done序列长度=transition数量，输入已经是前n个time_steps
        """
        transition_count = len(time_steps)
        dones = np.zeros(transition_count, dtype=bool)
        
        # 最后一个transition标记为done
        dones[-1] = True
        
        # 如果探索效率不再增长或视点全部访问，也可以标记为done
        # 这里暂时只使用最后一步done的简单逻辑
        
        return dones


class EPIC3DDatasetBuilder:
    """构建用于离线RL训练的数据集"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.processor = EPIC3DEpisodeProcessor(config)
        self.logger = logging.getLogger(__name__)
    
    def build_dataset_from_directories(self, data_dirs: List[str], output_path: str) -> str:
        """
        从多个episode目录构建数据集
        
        Args:
            data_dirs: 数据目录列表
            output_path: 输出HDF5文件路径
            
        Returns:
            str: 输出文件路径
        """
        all_episodes = []
        
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                self.logger.warning(f"Data directory does not exist: {data_dir}")
                continue
                
            episode_dirs = self._find_episode_directories(data_dir)
            self.logger.info(f"Found {len(episode_dirs)} episodes in {data_dir}")
            
            for episode_dir in episode_dirs:
                try:
                    self.logger.info(f"Processing episode: {episode_dir}")
                    episode_data = self.processor.process_episode_directory(episode_dir)
                    
                    # 检查episode是否完整（process_episode_directory可能返回None）
                    if episode_data is not None:
                        all_episodes.append(episode_data)
                        self.logger.info(f"Successfully processed episode: {episode_dir}")
                    else:
                        self.logger.warning(f"Episode discarded due to incompleteness: {episode_dir}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {episode_dir}: {e}")
                    continue
        
        if not all_episodes:
            raise ValueError("No valid episodes found in provided directories")
        
        self.logger.info(f"Processed {len(all_episodes)} episodes total")
        
        # 保存为HDF5格式
        self._save_to_hdf5(all_episodes, output_path)
        
        return output_path
    
    def _find_episode_directories(self, data_dir: str) -> List[str]:
        """查找episode目录"""
        episode_dirs = []
        
        try:
            # 检查当前目录是否直接包含filtered_data
            filtered_data_path = os.path.join(data_dir, 'filtered_data')
            if os.path.exists(filtered_data_path):
                # 当前目录本身就是episode目录
                episode_dirs.append(data_dir)
                self.logger.info(f"Found episode directory: {data_dir}")
            else:
                # 在子目录中查找episode目录
                for item in os.listdir(data_dir):
                    item_path = os.path.join(data_dir, item)
                    if os.path.isdir(item_path):
                        # 检查是否包含filtered_data目录
                        sub_filtered_data_path = os.path.join(item_path, 'filtered_data')
                        if os.path.exists(sub_filtered_data_path):
                            episode_dirs.append(item_path)
                            self.logger.info(f"Found episode directory: {item_path}")
        except Exception as e:
            self.logger.error(f"Failed to list directory {data_dir}: {e}")
        
        return sorted(episode_dirs)
    
    def _save_to_hdf5(self, episodes: List[Dict], output_path: str):
        """保存episodes到HDF5文件，直接输出graph_buffer兼容格式"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 按episode分别保存为独立批次文件（与旧版data_collector格式兼容）
        base_path = output_path.replace('.h5', '')
        batch_files = []
        
        for episode_idx, episode in enumerate(episodes):
            states = episode['states']
            
            # 转换为buffer兼容格式 - 处理维度重塑
            buffer_data = self._convert_to_buffer_format(states, episode)
            
            # 保存为独立批次文件
            batch_file_path = f"{base_path}_batch_{episode_idx+1}.h5"
            self._save_single_batch_file(batch_file_path, buffer_data, episode['metadata'])
            batch_files.append(batch_file_path)
        
        self.logger.info(f"数据集已保存为 {len(batch_files)} 个批次文件")
        for i, batch_file in enumerate(batch_files):
            self.logger.info(f"批次 {i+1}: {batch_file}")
        
        # 同时保存合并格式（用于兼容性）
        self._save_merged_format(episodes, output_path)

    def _convert_to_buffer_format(self, states: Dict, episode: Dict) -> Dict[str, np.ndarray]:
        """
        转换为buffer兼容格式，处理维度重塑问题
        
        重要修正：处理n+1个状态和n个transition的维度对齐
        - states包含n+1个观测
        - actions/rewards/dones只有n个
        - 需要裁剪状态到n个用于训练
        """
        T = len(episode['actions'])  # transition数量
        total_states = states['current_index'].shape[0]  # n+1个状态
        
        self.logger.debug(f"格式转换：{total_states}个状态 -> {T}个transition")
        
        # 只取前T个状态用于训练 (s_0到s_{T-1})
        # 注意：s_T作为最后一个transition的next_state将在训练时处理
        
        # 处理current_index维度重塑: (T,) → (T, 1, 1)
        current_index = states['current_index'][:T].reshape(T, 1, 1)
        
        # 处理viewpoints维度重塑: (T, max_viewpoints, 1) → (T, max_viewpoints, 1)
        viewpoints = states['viewpoints'][:T]
        if len(viewpoints.shape) == 2:
            viewpoints = viewpoints.reshape(T, viewpoints.shape[1], 1)
        
        # 处理node_padding_mask维度重塑: (T, 1, max_nodes) → (T, 1, max_nodes)
        node_padding_mask = states['node_padding_mask'][:T]
        if len(node_padding_mask.shape) == 2:
            node_padding_mask = node_padding_mask.reshape(T, 1, node_padding_mask.shape[1])
        
        # 处理viewpoint_padding_mask维度重塑: (T, 1, max_viewpoints) → (T, 1, max_viewpoints)  
        viewpoint_padding_mask = states['viewpoint_padding_mask'][:T]
        if len(viewpoint_padding_mask.shape) == 2:
            viewpoint_padding_mask = viewpoint_padding_mask.reshape(T, 1, viewpoint_padding_mask.shape[1])
        
        # 构建buffer格式数据
        buffer_data = {
            # 图结构数据 - 只取前T个状态
            'node_inputs': states['node_inputs'][:T].astype(np.float32),
            'adj_list': states['adj_list'][:T].astype(np.int64),
            
            # 重塑后的数据
            'node_padding_mask': node_padding_mask.astype(bool),
            'current_index': current_index.astype(np.int64),
            'viewpoints': viewpoints.astype(np.float32),
            'viewpoint_padding_mask': viewpoint_padding_mask.astype(bool),
            
            # RL数据 - 确保数据类型正确
            'actions': episode['actions'].astype(np.int64),
            'rewards': episode['rewards'].astype(np.float32), 
            'dones': episode['dones'].astype(bool),
        }
        
        return buffer_data
    
    def _save_single_batch_file(self, file_path: str, buffer_data: Dict[str, np.ndarray], metadata: Dict):
        """保存单个批次文件，格式与旧版data_collector完全兼容"""
        with h5py.File(file_path, 'w') as f:
            # 保存所有数据字段
            for key, data in buffer_data.items():
                f.create_dataset(key, data=data, compression='gzip', compression_opts=6)
            
            # 保存元数据 - 与旧版格式兼容
            f.attrs['episodes'] = 1  # 每个批次文件包含一个episode
            f.attrs['samples'] = len(buffer_data['actions'])
            
            # 保存维度信息
            if 'node_inputs' in buffer_data:
                f.attrs['node_dim'] = buffer_data['node_inputs'].shape[-1]
                f.attrs['num_nodes'] = buffer_data['node_inputs'].shape[1]
            
            # 添加兼容性标记
            f.attrs['format_version'] = 'epic3d_buffer_compatible'
            f.attrs['compatible_with'] = 'graph_buffer_v1'
            f.attrs['metadata'] = str(metadata)

    def _save_merged_format(self, episodes: List[Dict], output_path: str):
        """保存合并格式（用于兼容性和调试）"""
        # 合并所有episode的数据 - 使用原始格式
        all_node_inputs = []
        all_node_padding_mask = []
        all_current_index = []
        all_viewpoints = []
        all_viewpoint_padding_mask = []
        all_adj_list = []
        all_actions = []
        all_rewards = []
        all_dones = []
        
        episode_boundaries = [0]  # 记录每个episode的边界
        episode_metadata = []
        
        for episode in episodes:
            states = episode['states']
            episode_length = episode['episode_length']
            
            all_node_inputs.append(states['node_inputs'])
            all_node_padding_mask.append(states['node_padding_mask'])
            all_current_index.append(states['current_index'])
            all_viewpoints.append(states['viewpoints'])
            all_viewpoint_padding_mask.append(states['viewpoint_padding_mask'])
            all_adj_list.append(states['adj_list'])
            all_actions.append(episode['actions'])
            all_rewards.append(episode['rewards'])
            all_dones.append(episode['dones'])
            
            episode_boundaries.append(episode_boundaries[-1] + episode_length)
            episode_metadata.append(episode['metadata'])
        
        # 拼接所有数据
        try:
            combined_data = {
                'node_inputs': np.concatenate(all_node_inputs, axis=0),
                'node_padding_mask': np.concatenate(all_node_padding_mask, axis=0),
                'current_index': np.concatenate(all_current_index, axis=0),
                'viewpoints': np.concatenate(all_viewpoints, axis=0),
                'viewpoint_padding_mask': np.concatenate(all_viewpoint_padding_mask, axis=0),
                'adj_list': np.concatenate(all_adj_list, axis=0),
                'actions': np.concatenate(all_actions, axis=0),
                'rewards': np.concatenate(all_rewards, axis=0),
                'dones': np.concatenate(all_dones, axis=0)
            }
        except Exception as e:
            raise ValueError(f"Failed to concatenate episode data: {e}")
        
        # 保存到HDF5
        try:
            with h5py.File(output_path, 'w') as f:
                # 保存主要数据
                for key, data in combined_data.items():
                    f.create_dataset(key, data=data, compression='gzip', compression_opts=6)
                
                # 保存元数据
                f.attrs['num_episodes'] = len(episodes)
                f.attrs['total_samples'] = len(combined_data['actions'])
                f.attrs['episode_boundaries'] = np.array(episode_boundaries)
                f.attrs['config'] = str(self.config)
                f.attrs['creation_time'] = time.time()
                
                # 保存每个episode的元数据（作为字符串）
                metadata_strings = []
                for meta in episode_metadata:
                    metadata_strings.append(str(meta))
                
                f.create_dataset('episode_metadata', 
                               data=[s.encode('utf-8') for s in metadata_strings],
                               dtype=h5py.special_dtype(vlen=str))
                
        except Exception as e:
            raise ValueError(f"Failed to save HDF5 file {output_path}: {e}")
        
        self.logger.info(f"合并格式数据集已保存到: {output_path}")
        self.logger.info(f"总样本数: {len(combined_data['actions'])}")
        self.logger.info(f"总episodes: {len(episodes)}")
        
        # 打印数据集统计信息
        self._print_dataset_statistics(combined_data)
    
    def _print_dataset_statistics(self, combined_data: Dict):
        """打印数据集统计信息"""
        self.logger.info("=== Dataset Statistics ===")
        self.logger.info(f"Node inputs shape: {combined_data['node_inputs'].shape}")
        self.logger.info(f"Actions shape: {combined_data['actions'].shape}")
        self.logger.info(f"Rewards range: [{combined_data['rewards'].min():.3f}, {combined_data['rewards'].max():.3f}]")
        self.logger.info(f"Rewards mean: {combined_data['rewards'].mean():.3f}")
        self.logger.info(f"Actions unique values: {len(np.unique(combined_data['actions']))}")
        self.logger.info(f"Done episodes: {combined_data['dones'].sum()}")
        self.logger.info("=========================")


def setup_logging(level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


if __name__ == "__main__":
    # 简单测试
    setup_logging()
    
    config = {
        'max_nodes': 500,
        'max_viewpoints': 100,
        'k_size': 20,
        'node_feature_dim': 9,
        'position_scale': 100.0,
        'observation_score_scale': 50.0,
        'cluster_distance_scale': 20.0,
        'distance_scale': 50.0
    }
    
    # 测试单个文件解析
    parser = EPIC3DDataParser()
    test_file = "/home/amax/EPIC/collected_data/dungeon_batch_1_0_20250827_032058/filtered_data/topo_graph_1756236078.343951.txt"
    
    if os.path.exists(test_file):
        try:
            time_step = parser.parse_topo_file(test_file)
            print(f"Successfully parsed file with {len(time_step.nodes)} nodes and {len(time_step.edges)} edges")
            print(f"Viewpoints: {len(time_step.viewpoints)}")
            print(f"Current position: {time_step.current_pos}")
            print(f"Exploration stats: {time_step.exploration_stats}")
        except Exception as e:
            print(f"Failed to parse test file: {e}")
    else:
        print(f"Test file not found: {test_file}")
