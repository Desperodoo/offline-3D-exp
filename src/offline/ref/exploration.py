from multiprocessing import Pool, cpu_count
import os
import time, hydra, warnings, random, pickle
import logging
import imageio.v2 as imageio
import numpy as np
from numba import njit, typed, types
from copy import deepcopy

import ray
from .base_env import BaseEnv
from .Occupied_Grid_Map_Fixed import OccupiedGridMap
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning, NumbaPendingDeprecationWarning
import matplotlib.pyplot as plt
import matplotlib
from .graph_planner.build.devel.lib import graph_manager_py
import torch
from .utils import preprocess_cfg
# 导入新的可视化模块
from .visualization import snap_shot, gif_plot

logger = logging.getLogger(__name__)

warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)


class Exploration_Env(BaseEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.step_size = cfg.env.step_size
        self.d_min = cfg.graph.d_min
        self.d_max = cfg.graph.d_max
        self.voxel_size = cfg.graph.voxel_size
        self.resolution = cfg.map.resolution
        # 视点参数
        self.min_frontiers_num = cfg.vp.min_cluster_size
        self.max_cluster_radius = cfg.vp.max_cluster_radius
        self.candidate_rmin = cfg.vp.candidate_rmin
        self.candidate_rmax = cfg.vp.candidate_rmax
        self.candidate_rnum = int(cfg.vp.candidate_rnum)
        self.candidate_dphi = cfg.vp.candidate_dphi
        self.num_candidates = int(cfg.vp.num_candidates)
        self.NODE_PADDING_SIZE = cfg.graph.node_padding_size
        self.VIEWPOINT_PADDING_SIZE = cfg.graph.viewpoint_padding_size
        self.K_SIZE = cfg.graph.k_size
        self.device = cfg.algo.env_device
        self.map_size = cfg.map.width * cfg.map.height
        self.map_idx = random.choice(list(range(0, 500)))
        # self.map_idx = 38
        # Check if scale parameter exists, otherwise default to 'small'
        self.scale = getattr(cfg.map, 'scale', 'small')
        # Check if scene parameter exists, otherwise choose from options
        if hasattr(cfg.env, 'scene'):
            self.scene = cfg.env.scene
        else:
            self.scene = random.choice(['forest', 'office', 'tunnel'])
        self.cfg = cfg
        # Initialize graph manager with proper parameters
        self.graph_manager = graph_manager_py.GraphManager(
            self.resolution,                      # resolution
            cfg.map.width_grid,         # grid_num_x
            cfg.map.height_grid,        # grid_num_y
            cfg.map.inflation_radius,             # inflation_dist
            cfg.agent.sen_range,                     # max_range
            cfg.env.step_size,                    # step_size
            self.min_frontiers_num,               # min_frontiers_num
            self.max_cluster_radius,              # max_cluster_radius
            self.candidate_rmin,                  # candidate_rmin
            self.candidate_rmax,                  # candidate_rmax
            self.candidate_dphi,                  # candidate_dphi
            self.candidate_rnum,                  # candidate_rnum
            self.num_candidates                   # vpoints_per_cluster_num
        )
        
    def load_map(self):
        self.ground_truth.load('/home/lizh/simulation_env/map/map_0.pkl')
        
    def reset(self, save_img=False):
        self.time_step = 0
        self.n_episode += 1
        self.travel_dist = 0
        self.path = []
        self.current_target = None
        self.exploration_rate = 0
        self.delta_exp_rate = 0
        self.delta_dist = 0
        self.max_num_nodes = 0
        self.max_num_vps = 0
        self.max_k_size = 0
        # load map
        # self.init_map()  # grid_map, obstacles, boundary_map, boundary_obstacles, obstacle_agents, ex_grid_map, ex_obstacles, raser_map
        self.init_map_from_image(f'./Exploration_Env/map/{self.scale}/{self.scene}/{self.map_idx}.png')
        # print(f'Load map: {self.map_idx}, scale: {self.scale}, scene: {self.scene}')
        # self.init_map_from_image(f'./Exploration_Env/map/small/forest/-1.png')

        self.init_agent()
        self.trajectory = [[self.agent.x, self.agent.y]]
        self.eval_matrix = {
            'width': self.ground_truth.width_grid,
            'height': self.ground_truth.height_grid,
            'resolution': self.resolution,
            'pos': [],
            'path': [],
            'global_map': [],
            'local_map': [],
            'viewpoints': [],
            'clusters': [],
            'local_roadmap': [],
            'unknown_roadmap': [],
            'current_target': [],
            'merged_roadmap': [],
            'trajectory': [],
            'viewpoint_logits': [],
            'viewpoint_mask': [],
            'diffusion_logits': [],
            'map_dir': str(time.time())
        }
        
        observation, reward, done, info = self.get_observation(target_position=None, save_img=save_img)
        return observation, reward, done, info

    def reward(self, dist):
        reward = 0
        #  ================= DEBUG ==================== #
        # reward -= dist / self.map_size * 5  # 步数惩罚
        #  ================= DEBUG ==================== #
        reward -= dist * 0.05  # 步数惩罚
        #  ================= DEBUG ==================== #
        # reward += self.delta_exp_rate
        #  ================= DEBUG ==================== #
        return reward
    
    def get_observation(self, target_position=None, save_img=False):
        """
        Get the current observation, including node features, edge features, and viewpoint features.
        Returns (list): A list containing the following elements:
            node_inputs (torch.FloatTensor): Node features, shape (1, NODE_PADDING_SIZE, feature_dim).
            node_padding_mask (torch.Tensor): Node padding mask, shape (1, 1, NODE_PADDING_SIZE).
            edge_mask (torch.Tensor): Edge mask, shape (1, NODE_PADDING_SIZE, NODE_PADDING_SIZE).
            current_index (torch.Tensor): Current node index, shape (1, 1, 1).
            current_edge (torch.Tensor): Current edge index, shape (1, K_SIZE, 1).
            edge_padding_mask (torch.Tensor): Edge padding mask, shape (1, 1, K_SIZE).
            viewpoints (torch.Tensor): Viewpoint features, shape (1, VIEWPOINT_PADDING_SIZE, 1).
            viewpoint_padding_mask (torch.Tensor): Viewpoint padding mask, shape (1, 1, VIEWPOINT_PADDING_SIZE).
            adj_list (torch.Tensor): Adjacency list, shape (1, NODE_PADDING_SIZE, max_adj_list_size=K_SIZE).
        """
        start_time = time.time()
        # Graph generation/update
        t_start_graph = time.time()
        if target_position is None:
            current_pos = (self.agent.x, self.agent.y)
            global_grid = np.where(self.ground_truth.grid_map != 0, 1, 0).astype(np.int32)
            local_vertices, local_edges, vertex_dist, viewpoint_with_utilities, clusters, local_grid, current_pos, all_visited, trajectory, travel_dist = self.graph_manager.resetGraph(
                global_grid, self.d_min, self.d_max, self.voxel_size, current_pos)
        else:
            local_vertices, local_edges, vertex_dist, viewpoint_with_utilities, clusters, local_grid, current_pos, all_visited, trajectory, travel_dist = self.graph_manager.getGraph(self.d_min, self.d_max, self.voxel_size, target_position)
        t_graph = time.time() - t_start_graph

        t_start_validation = time.time()
        try:
            # 首先确保local_vertices不为空
            assert len(local_vertices) > 0, "local_vertices is empty, cannot find current_pos"
            assert len(local_vertices) < self.NODE_PADDING_SIZE, f'num of merged vertices: {len(local_vertices)} >= NODE_PADDING_SIZE: {self.NODE_PADDING_SIZE}'
            
            # 先尝试精确匹配
            try:
                current_index = np.where(np.all(np.array(local_vertices, dtype=np.float32) == np.array(current_pos, dtype=np.float32), axis=1))[0]
            except ValueError:  # 处理广播错误
                current_index = np.array([])
            
            # 如果精确匹配失败，尝试近似匹配
            if len(current_index) == 0:
                # print(f"Warning: current_pos {current_pos} not exactly found in local_vertices")
                # 使用find_point_index函数进行近似匹配，epsilon设为1e-4
                matched_idx = find_point_index(local_vertices, current_pos, epsilon=1e-3)
                
                if matched_idx >= 0:
                    # print(f"Found approximate match: {local_vertices[matched_idx]}")
                    current_index = np.array([matched_idx])
                else:
                    # 如果仍然找不到，尝试找最近的点
                    vertices_array = np.array(local_vertices, dtype=np.float32)
                    current_array = np.array(current_pos, dtype=np.float32)
                    if len(vertices_array) > 0:  # 确保vertices_array不为空
                        distances = np.sqrt(np.sum((vertices_array - current_array)**2, axis=1))
                        nearest_idx = np.argmin(distances)
                        min_distance = distances[nearest_idx]
                        
                        # 如果最近的点距离在可接受范围内
                        if min_distance < 0.1:  # 10cm以内的点视为可接受
                            # print(f"Using nearest vertex: {local_vertices[nearest_idx]} with distance {min_distance}")
                            current_index = np.array([nearest_idx])
                        else:
                            # print(f"Nearest vertex too far: {local_vertices[nearest_idx]} with distance {min_distance}")
                            raise AssertionError(f"current_pos {current_pos} not found in local_vertices (nearest distance: {min_distance})")
                    else:
                        raise AssertionError("local_vertices array conversion failed")
            
            current_idx = current_index[0]
            
            # 检查当前位置是否有邻居
            assert current_idx in local_edges and len(local_edges[current_idx]) > 0, "current_pos does not have at least one neighbor in merged_edges"
            
            assert len(viewpoint_with_utilities) < self.VIEWPOINT_PADDING_SIZE, f'num of viewpoints: {len(viewpoint_with_utilities)} >= VIEWPOINT_PADDING_SIZE: {self.VIEWPOINT_PADDING_SIZE}'
            
            # 检查trajectory的后几位，是否连续3个step都没有移动，可能是因为bug卡住了
            # Check if agent is stuck by analyzing recent trajectory positions
            if len(trajectory) > 3:
                last_steps = trajectory[-3:]
                # Use a small tolerance threshold when comparing positions
                tolerance = 1e-4  # Small threshold for position comparison
                is_stuck = True
                for step in last_steps[1:]:
                    # Check if positions are different beyond the tolerance threshold
                    if np.sqrt(np.sum((np.array(step) - np.array(last_steps[0]))**2)) > tolerance:
                        is_stuck = False
                        break
                if is_stuck:
                    raise AssertionError(f"Agent is stuck, last 3 steps are all at position {last_steps[0]}, target_position={target_position}, (within tolerance {tolerance})")
                
        except AssertionError as e:
            print(e)
            # self.snap_shot(self.ground_truth.width_grid, self.ground_truth.height_grid, current_pos, local_grid, viewpoint_with_utilities, clusters, local_vertices, local_edges, None, None, None, None, trajectory, str(e))
            if save_img:
                self.save_eval_matrix([current_pos, self.path, self.ground_truth.grid_map, local_grid, viewpoint_with_utilities, clusters, local_vertices, local_edges, None, None, target_position, None, None, trajectory])
            return None, None, None, None
        except Exception as e:
            print(f"Unexpected error during validation: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None
        t_validation = time.time() - t_start_validation

        # Feature preparation
        t_start_features = time.time()
        all_utility = []
        all_node_type = []  # 0: waypoint, 1: viewpoint, 2: frontier, 3: border, 4: unknown
        
        # Process viewpoints from viewpoint_with_utilities
        # viewpoint_with_utilities is a list of lists of tuples (x, y, utility)
        utility_dict = {}
        sum_utiliy = 0.0001  # Avoid division by zero
        
        # Flatten viewpoints and accumulate utilities
        for cluster_viewpoints in viewpoint_with_utilities:
            for vp in cluster_viewpoints:
                x, y, utility = vp
                vp_tuple = (x, y)
                if vp_tuple in utility_dict:
                    utility_dict[vp_tuple] += utility
                else:
                    utility_dict[vp_tuple] = utility
                sum_utiliy += utility
        
        # If the new format doesn't have viewpoints or utilities defined separately
        viewpoints = []
        utilities = []
        for vp_tuple, utility in utility_dict.items():
            viewpoints.append(vp_tuple)
            utilities.append(utility)

        for vertex in local_vertices:
            vertex_tuple = tuple(vertex)  # 将顶点转换为可哈希的类型
            all_utility.append(utility_dict.get(vertex_tuple, 0))  # 如果没有找到匹配的视点，添加 0
            all_node_type.append(0)
                
        node_coords = np.array(local_vertices)  # (n,2)所有点的坐标
        self.node_coords = node_coords
        node_utility = (np.array(all_utility)).reshape(-1, 1)   # utility
        node_guidepost = np.array(all_visited).reshape(-1, 1)
        node_type = np.array(all_node_type).reshape(-1, 1)
        node_dis = np.array(vertex_dist).reshape(-1, 1)
        # current_index = np.where(np.all(node_coords == current_pos, axis=1))[0]  # 当前索引 current_index = np.argwhere(node_coords == current_pos).reshape(-1)
        viewpoint_indices = []
        reachable_vp = []
        for vp in viewpoints:
            idx = np.where((node_coords == vp).all(axis=1))[0]
            if len(idx) >= 1:
                viewpoint_indices.append(idx[0])
                reachable_vp.append(vp)
            # else:
            #     print(f"viewpoint {vp} not in local_coords")
        t_features = time.time() - t_start_features

        # Save evaluation matrix if requested
        t_start_save = time.time()
        if save_img:
            self.save_eval_matrix([current_pos, self.path, self.ground_truth.grid_map, local_grid, viewpoint_with_utilities, clusters, local_vertices, local_edges, None, None, target_position, None, None, trajectory])
        t_save = time.time() - t_start_save

        # Tensor preparation
        t_start_tensor = time.time()
        # edge_mask = 1 - np.array(merged_edges)  # (n,n)邻接矩阵
        # current_edge = np.where(edge_mask[current_index] == 0)[1]  # 当前边索引 current_edge = np.argwhere(edge_mask[current_index] == 1).reshape(-1)

        n_node = node_coords.shape[0]
        if n_node > self.max_num_nodes:
            self.max_num_nodes = n_node
        current_node_coords = node_coords[current_index].squeeze()
        
        node_inputs = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                        node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                        axis=-1)/(np.sqrt((self.ground_truth.width_grid * self.resolution)**2 + (self.ground_truth.width_grid * self.resolution)**2))
        # node_inputs = np.concatenate((node_inputs, node_utility, node_guidepost), axis=-1)   # node features
        # node_inputs = np.concatenate((node_inputs, node_utility, node_guidepost, node_type), axis=-1)   # node features
        node_inputs = np.concatenate((node_inputs, node_utility, node_guidepost, node_type, node_dis), axis=-1)   # node features
        # node_inputs = np.concatenate((node_inputs, node_utility, node_guidepost, node_dis), axis=-1)   # node features

        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        padding = torch.nn.ZeroPad2d((0, 0, 0, self.NODE_PADDING_SIZE - n_node)) # 将节点数 padding 至 NODE_PADDING_SIZE
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, self.NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)    # *_padding_mask 指示那一部分有效

        current_index = torch.tensor(current_index).reshape(1, 1, 1).to(self.device)

        adj_list = np.full((n_node, self.K_SIZE), -1, dtype=int)

        # 直接从merged_edges填充adj_list
        if isinstance(local_edges, dict):
            # 邻接表格式直接使用
            for i in range(n_node):
                if i in local_edges:
                    valid_neighbors = [j for j in local_edges[i] if j < n_node]
                    if len(valid_neighbors) > self.max_k_size:
                        self.max_k_size = len(valid_neighbors)
                    num_neighbors = min(len(valid_neighbors), self.K_SIZE)
                    if num_neighbors > 0:
                        adj_list[i, :num_neighbors] = valid_neighbors[:num_neighbors]
        else:
            # 对于邻接矩阵，使用numpy高效操作
            for i in range(n_node):
                neighbors = np.where(local_edges[i, :n_node] == 1)[0]
                num_neighbors = min(len(neighbors), self.K_SIZE)
                if num_neighbors > 0:
                    adj_list[i, :num_neighbors] = neighbors[:num_neighbors]

        adj_list = torch.tensor(adj_list).unsqueeze(0).to(self.device)
        padding = torch.nn.ConstantPad2d((0, 0, 0, self.NODE_PADDING_SIZE - n_node), -1)
        adj_list = padding(adj_list)

        # edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)
        # padding = torch.nn.ConstantPad2d(
        #     (0, self.NODE_PADDING_SIZE - n_node, 0, self.NODE_PADDING_SIZE - n_node), 1)
        # edge_mask = padding(edge_mask)

        # current_edge = torch.tensor(current_edge).unsqueeze(0)
        # k_size = current_edge.size()[-1]
        # padding = torch.nn.ConstantPad1d((0, self.K_SIZE - k_size), 0)
        # current_edge = padding(current_edge)
        # current_edge = current_edge.unsqueeze(-1)

        # edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        # padding = torch.nn.ConstantPad1d((0, self.K_SIZE - k_size), 1)
        # edge_padding_mask = padding(edge_padding_mask)
        
        # 对 viewpoints 采取和 current_edge 相同的逻辑

        viewpoint_indices = torch.tensor(viewpoint_indices).unsqueeze(0).to(self.device)
        n_viewpoints = viewpoint_indices.size()[-1]
        if n_viewpoints > self.max_num_vps:
            self.max_num_vps = n_viewpoints
        padding = torch.nn.ConstantPad1d((0, self.VIEWPOINT_PADDING_SIZE - n_viewpoints), 0)
        viewpoint_indices = padding(viewpoint_indices)
        viewpoint_indices = viewpoint_indices.unsqueeze(-1)

        viewpoint_padding_mask = torch.zeros((1, 1, n_viewpoints), dtype=torch.int16).to(self.device)
        padding = torch.nn.ConstantPad1d((0, self.VIEWPOINT_PADDING_SIZE - n_viewpoints), 1)
        viewpoint_padding_mask = padding(viewpoint_padding_mask)
        t_tensor = time.time() - t_start_tensor
        
        observation = [
            node_inputs, 
            node_padding_mask, 
            None, # edge_mask 
            current_index, 
            None,  # current_edge 
            None,  # edge_padding_mask 
            viewpoint_indices, 
            viewpoint_padding_mask, 
            adj_list
        ]
        self.travel_dist += travel_dist
        reward = self.reward(dist=travel_dist)
        done = n_viewpoints == 0
        info = {'num_viewpoints': n_viewpoints, 'done:' : done, 'map_idx': self.map_idx}
        
        # Total time
        total_time = time.time() - start_time
        self.time_step += 1
        # # Print timing statistics
        # print("=== get_observation timing ===")
        # print(f"Graph generation: {t_graph:.4f}s ({(t_graph/total_time)*100:.1f}%)")
        # print(f"Validation: {t_validation:.4f}s ({(t_validation/total_time)*100:.1f}%)")
        # print(f"Feature preparation: {t_features:.4f}s ({(t_features/total_time)*100:.1f}%)")
        # print(f"Save matrices: {t_save:.4f}s ({(t_save/total_time)*100:.1f}%)")
        # print(f"Tensor preparation: {t_tensor:.4f}s ({(t_tensor/total_time)*100:.1f}%)")
        # print(f"Total time: {total_time:.4f}s (100%)")
        # print("=============================")
        
        return observation, reward, done, info

    def save_eval_matrix(self, eval_matrix):
        pos, path, global_map, local_map, viewpoints, clusters, local_vertices, local_edges, unknown_vertices, unknown_edges, current_target, merged_vertices, merged_edges, trajectory = eval_matrix
        self.eval_matrix['pos'].append(deepcopy(pos))
        self.eval_matrix['path'].append(deepcopy(path))
        self.eval_matrix['global_map'].append(deepcopy(global_map))
        self.eval_matrix['local_map'].append(deepcopy(local_map))
        self.eval_matrix['viewpoints'].append(deepcopy(viewpoints))
        self.eval_matrix['clusters'].append(deepcopy(clusters))
        self.eval_matrix['local_roadmap'].append([local_vertices, local_edges])
        self.eval_matrix['unknown_roadmap'].append([unknown_vertices, unknown_edges])
        self.eval_matrix['current_target'].append(deepcopy(current_target))
        self.eval_matrix['merged_roadmap'].append([merged_vertices, merged_edges])
        self.eval_matrix['trajectory'].append(deepcopy(trajectory))
        
    def save_eval_matrix_with_logits(self, eval_matrix, viewpoint_logits=None, viewpoint_mask=None, diffusion_logits=None):
        """保存包含logits信息的评估矩阵"""
        pos, path, global_map, local_map, viewpoints, clusters, local_vertices, local_edges, unknown_vertices, unknown_edges, current_target, merged_vertices, merged_edges, trajectory = eval_matrix
        self.eval_matrix['pos'].append(deepcopy(pos))
        self.eval_matrix['path'].append(deepcopy(path))
        self.eval_matrix['global_map'].append(deepcopy(global_map))
        self.eval_matrix['local_map'].append(deepcopy(local_map))
        self.eval_matrix['viewpoints'].append(deepcopy(viewpoints))
        self.eval_matrix['clusters'].append(deepcopy(clusters))
        self.eval_matrix['local_roadmap'].append([local_vertices, local_edges])
        self.eval_matrix['unknown_roadmap'].append([unknown_vertices, unknown_edges])
        self.eval_matrix['current_target'].append(deepcopy(current_target))
        self.eval_matrix['merged_roadmap'].append([merged_vertices, merged_edges])
        self.eval_matrix['trajectory'].append(deepcopy(trajectory))
        
        # 保存logits信息
        if viewpoint_logits is not None:
            self.eval_matrix['viewpoint_logits'].append(deepcopy(viewpoint_logits))
        if viewpoint_mask is not None:
            self.eval_matrix['viewpoint_mask'].append(deepcopy(viewpoint_mask))
        if diffusion_logits is not None:
            self.eval_matrix['diffusion_logits'].append(deepcopy(diffusion_logits))
    
    def gif_plot(self):
        matplotlib.use('Agg')
        gif_plot(num_frames=len(self.eval_matrix['pos']), params=self.eval_matrix, gif_dir=self.cfg.runner.gif_dir)
    
    def snap_shot(self, width, height, pos, local_map, viewpoints, clusters, local_vertices, local_edges, unknown_vertices, unknown_edges, merged_vertices, merged_edges, trajectory, debug_info,
                  viewpoint_logits=None, viewpoint_mask=None, diffusion_logits=None):
        snap_shot(width, height, pos, local_map, viewpoints, clusters, local_vertices, local_edges, unknown_vertices, unknown_edges, merged_vertices, merged_edges, trajectory, debug_info,
                  viewpoint_logits, viewpoint_mask, diffusion_logits)

    def update_eval_matrix_with_diffusion(self, diffusion_data):
        """更新eval_matrix中的扩散数据"""
        try:
            if 'viewpoint_logits' in diffusion_data:
                self.eval_matrix['viewpoint_logits'] = diffusion_data['viewpoint_logits']
            if 'viewpoint_mask' in diffusion_data:
                self.eval_matrix['viewpoint_mask'] = diffusion_data['viewpoint_mask']
            if 'diffusion_logits' in diffusion_data:
                self.eval_matrix['diffusion_logits'] = diffusion_data['diffusion_logits']
            
            logger.info(f"✓ Updated eval_matrix with diffusion data: {len(diffusion_data.get('viewpoint_logits', []))} steps")
        except Exception as e:
            logger.error(f"✗ Failed to update eval_matrix: {e}")
    
    def get_eval_matrix(self):
        """获取当前的eval_matrix"""
        return self.eval_matrix
    
    def set_eval_matrix_diffusion_data(self, diffusion_data):
        """设置eval_matrix的扩散数据"""
        self.update_eval_matrix_with_diffusion(diffusion_data)


def normalize_coordinates(vertices, precision=3):
    """
    高效规范化坐标列表的精度
    
    Args:
        vertices: 点列表 [(x1, y1), (x2, y2), ...] 或单个点 (x, y)
        precision: 保留小数位数
        
    Returns:
        规范化后的相同格式数据
    """
    if isinstance(vertices, tuple) and len(vertices) == 2:
        # 处理单个点 (x, y)
        # 处理单个点 (x, y)
        return tuple(np.round(np.array(vertices), precision).tolist())
    
    # 处理点列表，使用NumPy实现最高性能
    vertices_array = np.array(vertices)
    vertices_array = np.round(vertices_array, precision)
    return [tuple(point) for point in vertices_array]

def find_point_index(vertices, point, epsilon=1e-6):
    """
    查找点在列表中的索引，容忍浮点误差
    
    Args:
        vertices: 点列表 [(x1, y1), (x2, y2), ...]
        point: 要查找的点 (x, y)
        epsilon: 浮点比较的容差
        
    Returns:
        匹配的索引，如果未找到返回-1
    """
    for i, vertex in enumerate(vertices):
        if abs(vertex[0] - point[0]) < epsilon and abs(vertex[1] - point[1]) < epsilon:
            return i
    return -1


@ray.remote(num_cpus=0.5)
class RemoteEnv(Exploration_Env):
    def __init__(self, env_id, cfg, save_imgae=False):
        super().__init__(cfg)
        self.env_id = env_id
        self.save_image = save_imgae
        self.seed = random.randint(0, 1000000)
        # self.seed = 912902
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def _reset(self):
        while True:
            observation, reward, done, info = self.reset(save_img=self.save_image)
            if observation is not None and done is False:
                break
        return self.env_id, observation, self.node_coords, info

    def _step(self, target_position):
        observation, reward, done, info = self.get_observation(target_position=target_position, save_img=self.save_image)
        # if done:
        #     reward += 20
        if info is not None:
            info['seed'] = self.seed
        else:
            info = {}
            info['seed'] = self.seed
            print('info: ', info)
        return self.env_id, observation, self.node_coords, reward, done, info

    def _get_perf_metrics(self):
        perf_metrics = {}
        perf_metrics['travel_dist'] = self.travel_dist
        perf_metrics['time_steps'] = self.time_step
        perf_metrics['max_num_nodes'] = self.max_num_nodes
        perf_metrics['max_num_vps'] = self.max_num_vps
        perf_metrics['max_k_size'] = self.max_k_size
        perf_metrics['seed'] = self.seed
        perf_metrics['map'] = self.map_idx
        perf_metrics['scene'] = self.scene
        return self.env_id, perf_metrics
    
    def _get_eval_matrix(self):
        """获取当前的eval_matrix用于可视化"""
        return self.env_id, self.get_eval_matrix()
    
    def _plot(self):
        self.gif_plot()

