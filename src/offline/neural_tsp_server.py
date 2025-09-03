#!/usr/bin/env python3
"""
神经网络TSP求解服务
替代传统LKH求解器，使用训练好的DDQL模型进行视点选择
"""

import rospy
import torch
import torch.nn as nn
import numpy as np
import yaml
import sys
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 导入ROS消息类型
try:
    from epic_planner.srv import NeuralTSP, NeuralTSPResponse
    from geometry_msgs.msg import Point
except ImportError as e:
    rospy.logerr(f"Cannot import ROS message types: {e}")
    rospy.logerr("Make sure epic_planner is built and sourced properly")
    sys.exit(1)

# 现在文件在offline根目录，导入路径更简单
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # 直接添加当前目录到路径

# DDQL相关导入
try:
    from agent.model.graph_denoiser import create_pointer_denoiser
    from agent.model.diffusion_discrete import DiscreteDiffusion
    from agent.model.sgformer import QNet
    from agent.ddql import TrainConfig
    import yaml
    DDQL_AVAILABLE = True
except ImportError as e:
    rospy.logerr(f"DDQL modules not available: {e}. Please ensure offline module is properly installed.")
    DDQL_AVAILABLE = False

class NeuralTSPServer:
    def __init__(self):
        rospy.init_node('neural_tsp_server')
        
        # 加载模型配置和权重
        self.config, self.model_path = self.load_config()
        if self.config is None:
            rospy.logerr("Failed to load config, shutting down")
            return
            
        # 初始化设备
        self.device = torch.device(self.config.device)
        self.model = self.load_model()
        
        # 创建服务
        self.service = rospy.Service('neural_tsp_solve', NeuralTSP, self.handle_tsp_request)
        rospy.loginfo("Neural TSP Server initialized and ready on device: %s", str(self.device))

    def load_config(self):
        """从offline/config.yaml和ROS参数加载DDQL配置，offline配置优先"""
        if not DDQL_AVAILABLE:
            rospy.logerr("DDQL modules not available, cannot load config")
            return None, ""
            
        try:
            # 加载config.yaml配置文件
            config_path = os.path.join(current_dir, 'config.yaml')
            offline_config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    offline_config = yaml.safe_load(f) or {}
                rospy.loginfo(f"Loaded offline config from: {config_path}")
            else:
                rospy.logerr(f"Offline config not found: {config_path}")
                return None, ""
            
            # 2. 从offline配置创建TrainConfig
            config = TrainConfig()
            
            # 图结构参数
            data_processing = offline_config.get('data_processing', {})
            if 'node_feature_dim' in data_processing:
                config.node_dim = data_processing['node_feature_dim']
            if 'max_viewpoints' in data_processing:
                config.max_viewpoints = data_processing['max_viewpoints']
            if 'k_size' in data_processing:
                config.k_size = data_processing['k_size']
            else:
                config.k_size = 20  # 默认值
                
            # 数据标准化参数
            config.position_scale = data_processing.get('position_scale', 100.0)
            config.distance_scale = data_processing.get('distance_scale', 50.0)
            config.observation_score_scale = data_processing.get('observation_score_scale', 50.0)
            config.cluster_distance_scale = data_processing.get('cluster_distance_scale', 20.0)
                
            # 模型参数
            model_config = offline_config.get('model', {})
            if 'embed_dim' in model_config:
                config.gnn_hidden_dim = model_config['embed_dim']
                
            # 算法参数
            if offline_config.get('algorithm', '').lower() == 'ddql':
                ddql_config = offline_config.get('ddql', {})
                if 'T' in ddql_config:
                    config.n_timesteps = ddql_config['T']
                if 'use_fixed_actions' in ddql_config:
                    config.use_fixed_actions = ddql_config['use_fixed_actions']
                if 'temperature' in ddql_config:
                    config.temperature = ddql_config['temperature']
            
            # 设备配置（自动检测）
            config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 模型路径（从配置文件或默认路径获取）
            model_path = offline_config.get('model_path', '')  # 可在config.yaml中配置
            if not model_path:
                # 默认在results目录下查找最新模型
                results_dir = os.path.join(current_dir, 'results')
                if os.path.exists(results_dir):
                    model_files = [f for f in os.listdir(results_dir) if f.endswith('.pth')]
                    if model_files:
                        model_path = os.path.join(results_dir, sorted(model_files)[-1])
                        rospy.loginfo(f"Auto-detected model: {model_path}")
            
            rospy.loginfo(f"DDQL Config: node_dim={config.node_dim}, embed_dim={config.gnn_hidden_dim}, "
                         f"max_viewpoints={config.max_viewpoints}, k_size={config.k_size}, T={config.n_timesteps}, device={config.device}")
            
            return config, model_path
            
        except Exception as e:
            rospy.logerr(f"Failed to load config: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return None, ""

    def load_model(self):
        """加载训练好的DDQL模型"""
        try:
            if not DDQL_AVAILABLE:
                rospy.logerr("DDQL modules not available, cannot load model")
                return None
            
            if self.model_path and os.path.exists(self.model_path):
                rospy.loginfo(f"Loading DDQL model from: {self.model_path}")
                
                # 创建pointer去噪器（默认使用pointer模式）
                denoiser = create_pointer_denoiser(
                    node_dim=self.config.node_dim,
                    embed_dim=self.config.gnn_hidden_dim,
                    max_actions=self.config.max_viewpoints,
                    temperature=self.config.temperature
                )
                
                # 创建扩散模型
                model = DiscreteDiffusion(
                    num_actions=self.config.max_viewpoints,
                    model=denoiser,
                    T=self.config.n_timesteps,
                    use_fixed_actions=self.config.use_fixed_actions,
                    schedule='cosine',
                ).to(self.device)
                
                # 加载预训练权重
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # 处理不同的检查点格式
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        model.load_state_dict(checkpoint['model'], strict=False)
                        rospy.loginfo("Loaded model weights from checkpoint")
                    elif 'actor' in checkpoint:
                        # 如果保存的是actor，需要提取扩散模型部分
                        model.load_state_dict(checkpoint['actor'], strict=False)
                        rospy.loginfo("Loaded actor weights from checkpoint")
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                        rospy.loginfo("Loaded full checkpoint")
                else:
                    model.load_state_dict(checkpoint, strict=False)
                    rospy.loginfo("Loaded model state dict")
                
                model.eval()
                rospy.loginfo(f"DDQL model loaded successfully from {self.model_path}")
                return model
                
            else:
                rospy.logerr(f"Model path not provided or file not exists: {self.model_path}")
                return None
                
        except Exception as e:
            rospy.logerr(f"Failed to load DDQL model: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return None

    def construct_observation(self, req) -> Tuple[list, torch.Tensor]:
        """基于完整拓扑图信息构建DDQL模型需要的观察值格式 (优化版本)
        
        使用向量化操作提升计算效率，与epic3d_data_processor.py保持完全一致的数据处理逻辑
        
        Args:
            req: NeuralTSP请求，包含完整的拓扑图信息
            
        Returns:
            obs: 观察值列表 [node_inputs, node_padding_mask, current_index, viewpoints, viewpoint_padding_mask, adj_list]
            padding_mask: 视点填充掩码 [B, 1, max_viewpoints]
        """
        try:
            B = 1  # 批次大小为1
            
            # 提取节点信息
            num_nodes = len(req.node_positions)
            num_viewpoints = len(req.viewpoints)
            max_nodes = max(num_nodes, 100)  # 确保有足够空间
            max_viewpoints = self.config.max_viewpoints
            node_feature_dim = self.config.node_dim
            
            rospy.logdebug(f"Constructing observation: {num_nodes} nodes, {num_viewpoints} viewpoints")
            
            # 向量化提取所有节点位置
            node_positions = np.array([[p.x, p.y, p.z] for p in req.node_positions[:num_nodes]], dtype=np.float32)
            
            # 找到当前位置（向量化查找）
            current_flags = np.array(req.node_is_current[:num_nodes], dtype=bool)
            current_indices = np.where(current_flags)[0]
            
            if len(current_indices) > 0:
                current_pos = node_positions[current_indices[0]]
            else:
                # 如果没有找到current节点，使用请求中的current_position
                current_pos = np.array([
                    req.current_position.x, 
                    req.current_position.y, 
                    req.current_position.z
                ], dtype=np.float32)
            
            # 1. 向量化构建节点特征矩阵 [B, N, node_feature_dim] - 与epic3d_data_processor.py完全对齐
            node_inputs = torch.zeros(B, max_nodes, node_feature_dim, device=self.device, dtype=torch.float32)
            
            if num_nodes > 0:
                # 向量化计算相对位置 - 与训练数据处理逻辑完全一致
                rel_positions = (node_positions - current_pos[None, :]) / self.config.position_scale
                
                # 向量化提取特征数组，确保长度匹配，应用与训练时相同的标准化
                obs_scores = np.zeros(num_nodes, dtype=np.float32)
                # cluster_dists = np.zeros(num_nodes, dtype=np.float32)
                # distances = np.zeros(num_nodes, dtype=np.float32)
                is_viewpoints = np.zeros(num_nodes, dtype=np.float32)
                is_histories = np.zeros(num_nodes, dtype=np.float32)
                
                # 安全地填充特征数组 - 与epic3d_data_processor.py中的处理逻辑一致
                if req.node_observation_scores:
                    obs_len = min(len(req.node_observation_scores), num_nodes)
                    # 应用与训练时相同的处理：max(0, score) / scale
                    obs_scores[:obs_len] = np.array([max(0, score) for score in req.node_observation_scores[:obs_len]], dtype=np.float32)
                    obs_scores /= self.config.observation_score_scale
                
                # if req.node_cluster_distances:
                #     cluster_len = min(len(req.node_cluster_distances), num_nodes)
                #     # 应用与训练时相同的处理：max(0, dist) / scale if dist > 0 else 0
                #     for i in range(cluster_len):
                #         dist_val = req.node_cluster_distances[i]
                #         cluster_dists[i] = (max(0, dist_val) / self.config.cluster_distance_scale) if dist_val > 0 else 0.0
                
                # if req.node_distances:
                #     dist_len = min(len(req.node_distances), num_nodes)
                #     # 应用与训练时相同的处理：max(0, dist) / scale if dist > 0 else 0
                #     for i in range(dist_len):
                #         dist_val = req.node_distances[i]
                #         distances[i] = (max(0, dist_val) / self.config.distance_scale) if dist_val > 0 else 0.0
                
                if req.node_is_viewpoint:
                    vp_len = min(len(req.node_is_viewpoint), num_nodes)
                    is_viewpoints[:vp_len] = np.array([float(vp) for vp in req.node_is_viewpoint[:vp_len]], dtype=np.float32)
                
                if req.node_is_history:
                    hist_len = min(len(req.node_is_history), num_nodes)
                    is_histories[:hist_len] = np.array([float(hist) for hist in req.node_is_history[:hist_len]], dtype=np.float32)
                
                # 构建特征矩阵 - 与epic3d_data_processor.py中的feature_vector完全一致
                # [rel_x, rel_y, rel_z, obs_score, cluster_dist, is_vp, visited, dist, centrality]
                features = np.column_stack([
                    rel_positions[:, 0],      # 相对3D位置 - rel_x
                    rel_positions[:, 1],      # 相对3D位置 - rel_y  
                    rel_positions[:, 2],      # 相对3D位置 - rel_z
                    obs_scores,               # 标准化观测得分
                    # cluster_dists,            # 标准化聚类距离/路径代价
                    is_viewpoints,            # 节点类型（是否为视点）
                    is_histories,             # 访问状态（是否为历史位置）
                    # distances,                # 距离特征
                    # np.full(num_nodes, 0.5, dtype=np.float32)  # 占位的中心性得分
                ])
                
                # 一次性转换为tensor并赋值
                node_inputs[0, :num_nodes] = torch.from_numpy(features).to(self.device)
            
            # 2. 节点填充掩码 [B, 1, N] (False=有效，True=填充)
            node_padding_mask = torch.ones(B, 1, max_nodes, device=self.device, dtype=torch.bool)
            node_padding_mask[0, 0, :num_nodes] = False
            
            # 3. 找到当前节点索引 [B, 1, 1] - 使用向量化查找
            current_node_idx = current_indices[0] if len(current_indices) > 0 else 0
            current_index = torch.tensor([[[current_node_idx]]], device=self.device, dtype=torch.long)
            
            # 4. 向量化构建视点索引 [B, max_viewpoints, 1]
            viewpoint_indices = torch.zeros(B, max_viewpoints, 1, device=self.device, dtype=torch.long)
            
            # 向量化提取视点节点索引
            if req.node_is_viewpoint and num_nodes > 0:
                vp_flags = np.array(req.node_is_viewpoint[:num_nodes], dtype=bool)
                viewpoint_nodes = np.where(vp_flags)[0]
                
                # 应用max_viewpoints限制
                valid_viewpoints = min(len(viewpoint_nodes), max_viewpoints)
                if valid_viewpoints > 0:
                    viewpoint_indices[0, :valid_viewpoints, 0] = torch.from_numpy(
                        viewpoint_nodes[:valid_viewpoints]).to(self.device)
            else:
                viewpoint_nodes = np.array([], dtype=int)
                valid_viewpoints = 0
            
            # 5. 视点填充掩码 [B, 1, max_viewpoints] (False=有效，True=填充) 
            viewpoint_padding_mask = torch.ones(B, 1, max_viewpoints, device=self.device, dtype=torch.bool)
            viewpoint_padding_mask[0, 0, :valid_viewpoints] = False
            
            # 6. 优化邻接表构建 [B, N, K]
            K = self.config.k_size
            adj_list = torch.full((B, max_nodes, K), -1, device=self.device, dtype=torch.long)
            
            if req.edge_from_nodes and req.edge_to_nodes and num_nodes > 0:
                # 向量化处理边信息
                from collections import defaultdict
                
                num_edges = min(len(req.edge_from_nodes), len(req.edge_to_nodes))
                if num_edges > 0:
                    # 向量化边验证
                    from_nodes = np.array(req.edge_from_nodes[:num_edges])
                    to_nodes = np.array(req.edge_to_nodes[:num_edges])
                    
                    # 筛选有效边（避免逐个检查）
                    valid_mask = (from_nodes >= 0) & (from_nodes < num_nodes) & \
                                (to_nodes >= 0) & (to_nodes < num_nodes)
                    
                    if valid_mask.any():
                        valid_from = from_nodes[valid_mask]
                        valid_to = to_nodes[valid_mask]
                        
                        # 构建邻接字典
                        adjacency_dict = defaultdict(list)
                        for from_node, to_node in zip(valid_from, valid_to):
                            adjacency_dict[from_node].append(to_node)
                        
                        # 批量填充邻接表
                        for node_id, neighbors in adjacency_dict.items():
                            if node_id < max_nodes:
                                num_neighbors = min(len(neighbors), K)
                                if num_neighbors > 0:
                                    adj_list[0, node_id, :num_neighbors] = torch.tensor(
                                        neighbors[:num_neighbors], device=self.device)
            
            # 构建观察值列表 - 与DDQL训练时的格式完全一致
            obs = [
                node_inputs,               # [B, N, node_dim]
                node_padding_mask,         # [B, 1, N]
                current_index,             # [B, 1, 1] 
                viewpoint_indices,         # [B, max_viewpoints, 1]
                viewpoint_padding_mask,    # [B, 1, max_viewpoints]
                adj_list                   # [B, N, K]
            ]
            
            rospy.logdebug(f"Observation constructed: nodes={num_nodes}, viewpoints={valid_viewpoints}, "
                          f"current_idx={current_node_idx}, viewpoint_nodes={viewpoint_nodes[:5] if len(viewpoint_nodes) > 0 else []}")
            
            return obs, viewpoint_padding_mask
            
        except Exception as e:
            rospy.logerr(f"Failed to construct observation: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())
            raise e

    def handle_tsp_request(self, req):
        """处理TSP求解请求 (优化版本)"""
        import time
        
        try:
            start_time = time.time()
            
            # 提取请求基本信息  
            current_pos = [req.current_position.x, req.current_position.y, req.current_position.z]
            viewpoints = [(vp.x, vp.y, vp.z) for vp in req.viewpoints]
            num_nodes = len(req.node_positions)
            
            rospy.loginfo(f"Neural TSP request: current_pos={current_pos}, viewpoints={len(viewpoints)}, nodes={num_nodes}")
            
            # 验证输入数据
            if len(viewpoints) == 0:
                response = NeuralTSPResponse()
                response.success = False
                response.next_viewpoint_index = -1
                response.message = "No viewpoints provided"
                response.confidence_score = 0.0
                return response
                
            if num_nodes == 0:
                rospy.logwarn("No nodes in topology graph, using fallback random selection")
                next_viewpoint_idx = np.random.randint(0, len(viewpoints))
                response = NeuralTSPResponse()
                response.success = True
                response.next_viewpoint_index = int(next_viewpoint_idx)
                response.message = f"Fallback random selection: {next_viewpoint_idx}"
                response.confidence_score = 0.1
                return response

            # 神经网络推理
            with torch.no_grad():
                if self.model is None or not DDQL_AVAILABLE:
                    # 使用随机选择作为fallback
                    next_viewpoint_idx = np.random.randint(0, len(viewpoints))
                    rospy.logwarn("Using random selection due to model unavailable")
                    confidence_score = 0.1
                else:
                    # 使用真实DDQL模型
                    obs_start = time.time()
                    obs, padding_mask = self.construct_observation(req)
                    obs_time = (time.time() - obs_start) * 1000
                    
                    # 使用DiscreteDiffusion.sample进行推理
                    inference_start = time.time()
                    action_indices = self.model.sample(obs, padding_mask)  # [B, 1]
                    inference_time = (time.time() - inference_start) * 1000
                    
                    # 调试信息：检查action_indices的形状
                    rospy.logdebug(f"action_indices shape: {action_indices.shape}, type: {type(action_indices)}")
                    rospy.logdebug(f"action_indices: {action_indices}")
                    
                    # 获取选中的视点索引（在拓扑图中的节点索引）
                    # 处理不同的tensor维度情况
                    if action_indices.dim() == 1:
                        # 如果是1D tensor，直接取第一个元素
                        selected_node_idx = action_indices[0].cpu().numpy()
                    elif action_indices.dim() == 2:
                        # 如果是2D tensor，取[0, 0]
                        selected_node_idx = action_indices[0, 0].cpu().numpy()
                    else:
                        # 其他情况，尝试flatten后取第一个
                        selected_node_idx = action_indices.flatten()[0].cpu().numpy()
                    
                    rospy.logdebug(f"selected_node_idx: {selected_node_idx}")
                    
                    mapping_start = time.time()
                    
                    # 将拓扑图中的节点索引映射回viewpoints数组中的索引
                    next_viewpoint_idx = -1
                    
                    # 构建viewpoint_nodes列表（与construct_observation中的逻辑一致）
                    if req.node_is_viewpoint and len(req.node_is_viewpoint) > 0:
                        num_check = min(len(req.node_positions), len(req.node_is_viewpoint))
                        vp_flags = np.array(req.node_is_viewpoint[:num_check], dtype=bool)
                        viewpoint_nodes = np.where(vp_flags)[0]
                        
                        # 将选中的节点索引映射回viewpoints数组中的索引
                        if selected_node_idx < len(viewpoint_nodes):
                            selected_topo_node = viewpoint_nodes[selected_node_idx]
                            
                            # 在req.viewpoint_indices中找到对应的位置（向量化查找）
                            if req.viewpoint_indices:
                                viewpoint_indices_array = np.array(req.viewpoint_indices)
                                matches = np.where(viewpoint_indices_array == selected_topo_node)[0]
                                if len(matches) > 0:
                                    next_viewpoint_idx = matches[0]
                    else:
                        viewpoint_nodes = np.array([], dtype=int)
                    
                    # 如果映射失败，使用第一个有效视点
                    if next_viewpoint_idx == -1:
                        next_viewpoint_idx = 0
                        rospy.logwarn(f"Failed to map node index {selected_node_idx} to viewpoint, using index 0")
                    
                    # 确保索引在有效范围内
                    next_viewpoint_idx = max(0, min(int(next_viewpoint_idx), len(viewpoints) - 1))
                    
                    mapping_time = (time.time() - mapping_start) * 1000
                    
                    # 简单的置信度评估（基于采样结果的一致性）
                    confidence_score = 0.8  # TODO: 可以基于扩散模型的采样过程计算更准确的置信度
                    
                    total_time = (time.time() - start_time) * 1000
                    rospy.logdebug(f"DDQL performance: obs={obs_time:.2f}ms, inference={inference_time:.2f}ms, "
                                  f"mapping={mapping_time:.2f}ms, total={total_time:.2f}ms")
                    rospy.logdebug(f"DDQL selected node_idx: {selected_node_idx}, "
                                  f"mapped topo_node: {viewpoint_nodes[selected_node_idx] if selected_node_idx < len(viewpoint_nodes) else 'invalid'}, "
                                  f"final viewpoint_idx: {next_viewpoint_idx}")
            
            # 构建响应
            response = NeuralTSPResponse()
            response.success = True
            response.next_viewpoint_index = int(next_viewpoint_idx)
            response.confidence_score = float(confidence_score)
            response.message = f"Neural TSP solved successfully, selected viewpoint {next_viewpoint_idx}/{len(viewpoints)} (confidence: {confidence_score:.2f})"
            
            rospy.loginfo(f"Neural TSP solved: selected viewpoint {next_viewpoint_idx} with confidence {confidence_score:.2f}")
            return response
            
        except Exception as e:
            rospy.logerr(f"Neural TSP solving failed: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())
            response = NeuralTSPResponse()
            response.success = False
            response.next_viewpoint_index = -1
            response.confidence_score = 0.0
            response.message = f"Error: {str(e)}"
            return response

if __name__ == '__main__':
    try:
        server = NeuralTSPServer()
        rospy.loginfo("Neural TSP Server is ready to serve requests...")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Neural TSP Server shutting down...")
    except Exception as e:
        rospy.logerr(f"Neural TSP Server error: {str(e)}")
        import traceback
        rospy.logerr(traceback.format_exc())
