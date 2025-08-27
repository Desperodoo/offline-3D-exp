from copy import deepcopy
import random
from omegaconf import OmegaConf
import torch
from collections import OrderedDict
import hydra
import time
import rospy
import logging
import torch.multiprocessing as mp
import os
import numpy as np
import ray
from Exploration_Env.utils import preprocess_cfg
from Exploration_Env.exploration import RemoteEnv
# from model.ddql_graph_discrete import initialize_ddql, TrainConfig as DDQLTrainConfig
from ms_latent.ddql_graph_discrete import initialize_multistep_latent_ddql as initialize_ddql, MultiStepLatentTrainConfig as DDQLTrainConfig
from Exploration_Env.visualization import visualize_diffusion_process


@ray.remote(num_cpus=0.5)
class Worker:
    def __init__(self, cfg, algo_cfg, algo_type, meta_agent_id, local_rank=0, save_image=False):
        """
        Worker类，支持DDQL算法
        
        参数:
            cfg: 环境配置
            algo_cfg: 算法配置
            algo_type: 算法类型，必须是"ddql"
            meta_agent_id: Worker ID
            local_rank: 本地排名
            save_image: 是否保存图像
        """
        self.meta_agent_id = meta_agent_id
        self.algo_type = algo_type.lower()
        self.resolution = cfg.map.resolution
        node_dim = cfg.graph.node_input_dim
        self.worker_device = torch.device("cpu")

        # 初始化DDQL模型
        if self.algo_type == "ddql":
            model = initialize_ddql(config=algo_cfg, node_dim=node_dim, rank=0, world_size=1)
            self.policy = model.actor
        else:
            raise ValueError(f"不支持的算法类型: {algo_type}，只支持DDQL")
        
        self.buffer_device = cfg.algo.env_device
        self.cfg = cfg
        self.num_envs = cfg.runner.num_envs_per_worker
        self.save_image = save_image

    def init(self, state_dict):
        if type(state_dict) == OrderedDict or type(state_dict) == dict:
            self.policy.load_state_dict(state_dict)
        elif state_dict is None:
            pass
        else:
            raise ValueError("Invalid state_dict type")

        self.perf_metrics = dict()
        self.init_buffer()
        # 每个 env 单独维护一个指针
        self.env_pointers = [0] * self.num_envs
        
    def init_buffer(self):
        max_step = self.cfg.env.max_steps
        node_padding_size = self.cfg.graph.node_padding_size
        viewpoint_padding_size = self.cfg.graph.viewpoint_padding_size
        node_input_dim = self.cfg.graph.node_input_dim
        k_size = self.cfg.graph.k_size
        self.episode_buffer = dict()
        # 保持图结构相关字段不变
        # self.episode_buffer['node_inputs'] = torch.empty((self.num_envs, max_step, node_padding_size, node_input_dim), dtype=torch.float32, device=self.buffer_device)
        # self.episode_buffer['node_padding_mask'] = torch.empty((self.num_envs, max_step, 1, node_padding_size), dtype=torch.bool, device=self.buffer_device)
        # self.episode_buffer['current_index'] = torch.empty((self.num_envs, max_step, 1, 1), dtype=torch.int64, device=self.buffer_device)
        # self.episode_buffer['viewpoints'] = torch.empty((self.num_envs, max_step, viewpoint_padding_size, 1), dtype=torch.float32, device=self.buffer_device)
        # self.episode_buffer['viewpoint_padding_mask'] = torch.empty((self.num_envs, max_step, 1, viewpoint_padding_size), dtype=torch.bool, device=self.buffer_device)
        # self.episode_buffer['adj_list'] = torch.empty((self.num_envs, max_step, node_padding_size, k_size), dtype=torch.int64, device=self.buffer_device)

        # 更新字段名称，使用统一的复数形式
        # self.episode_buffer['logp'] = torch.empty((self.num_envs, max_step), dtype=torch.float32, device=self.buffer_device)
        # self.episode_buffer['actions'] = torch.empty((self.num_envs, max_step), dtype=torch.int64, device=self.buffer_device)
        self.episode_buffer['rewards'] = torch.empty((self.num_envs, max_step), dtype=torch.float32, device=self.buffer_device)
        self.episode_buffer['dones'] = torch.empty((self.num_envs, max_step), dtype=torch.int8, device=self.buffer_device)

    def choose_viewpoint(self, observation, node_coords, percentage=None, num_inference_steps=None):
        """
        统一的视点选择接口，根据算法类型调用不同的推理方法
        
        Args:
            observation: 环境观测数据
            node_coords: 节点坐标
            percentage: 预留参数（某些算法使用）
            num_inference_steps: 自定义扩散推理步数（仅DDQL算法使用）
        """
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, viewpoints, viewpoint_padding_mask, adj_list = observation
        mini_observation = [node_inputs, node_padding_mask, current_index, viewpoints, viewpoint_padding_mask, adj_list]
        mini_observation = [mini_observation[i].to(self.worker_device) for i in range(len(mini_observation))]
        
        viewpoints = mini_observation[3]
        
        # 根据算法类型执行不同的推理
        with torch.no_grad():

            # action_index = self.policy.sample(
            #     obs=mini_observation, 
            #     padding_mask=mini_observation[4], 
            #     return_intermediate=False
            # )
            
            # 🎯 支持自定义推理步数的DDQL预测
            predict_results = self.policy.predict(mini_observation, deterministic=False, num_inference_steps=num_inference_steps)
            action_index = predict_results['actions']

        next_node_index = viewpoints[0, action_index.item(), 0].item()
        try:
            next_position = node_coords[next_node_index]
        except:
            print(f"next_node_index: {next_node_index}, node_coords: {node_coords}")
            print(f"viewpoints: {viewpoints}")
            raise ValueError
        
        # DDQL不返回logp，设为0
        logp = torch.tensor(0.0)
        
        return next_position, action_index, logp

    def run_rollout(self, percentage=None, num_inference_steps=None):
        """
        执行rollout，使用DDQL算法
        
        参数:
            percentage: 预留参数，DDQL算法忽略此参数
            num_inference_steps: 自定义扩散推理步数（仅DDQL算法使用）
        """
        # 初始化计时统计
        predict_time = 0.0
        env_step_times = {}  # 记录每个env step的开始时间
        total_env_step_time = 0.0
        step_count = 0
        reset_time = 0.0
        
        # 初始化环境
        self.envs = [RemoteEnv.remote(i, self.cfg, self.save_image) for i in range(self.num_envs)]
        valid_envs = [True] * self.num_envs

        # 重置环境
        reset_start = time.time()
        reset_refs = [self.envs[i]._reset.remote() for i in range(self.num_envs)]
        refs = []
        success_times = 0

        # 处理重置阶段
        while len(reset_refs) > 0:
            done_ref, reset_refs = ray.wait(reset_refs, num_returns=1)
            env_id, observation, node_coords, info = ray.get(done_ref[0])
            
            # 记录模型推理时间
            predict_start = time.time()
            target_position, action_index, logp = self.choose_viewpoint(observation, node_coords, percentage, num_inference_steps)
            # self.save_observation(env_id, observation)
            # self.save_action(env_id, action_index, logp)
            predict_time += time.time() - predict_start

            # 记录环境step的开始时间
            step_ref = self.envs[env_id]._step.remote(target_position=target_position)
            env_step_times[step_ref] = time.time()
            refs.append(step_ref)
            step_count += 1

        reset_time = time.time() - reset_start
        
        # 主循环
        while len(refs) > 0:
            done_ref, refs = ray.wait(refs, num_returns=1)

            # 记录环境step完成时间
            if env_step_times:
                step_end_time = time.time()
                for ref_id in env_step_times:
                    if ref_id == done_ref[0]:
                        step_duration = step_end_time - env_step_times[ref_id]
                        total_env_step_time += step_duration
                        del env_step_times[ref_id]
                        break
            
            env_id, observation, node_coords, reward, done, info = ray.get(done_ref[0])
            
            if (done is None) or (self.env_pointers[env_id] >= self.cfg.env.max_steps - 1):
                valid_envs[env_id] = False
                if done is None:
                    print(f"Env {env_id} failed for some reason!")
                else:
                    print(f"Env {env_id} exceeded the maximum number of steps.")
                    print(f"info: {info}")

            elif done is False:
                # 记录模型推理时间
                predict_start = time.time()
                target_position, action_index, logp = self.choose_viewpoint(observation, node_coords, percentage, num_inference_steps)
                self.save_reward(env_id, reward)
                self.save_done(env_id, done)
                # self.save_observation(env_id, observation)
                # self.save_action(env_id, action_index, logp)
                predict_time += time.time() - predict_start
                
                # 记录环境step的开始时间
                step_ref = self.envs[env_id]._step.remote(target_position=target_position)
                env_step_times[step_ref] = time.time()
                refs.append(step_ref)
                step_count += 1
            else:
                self.save_reward(env_id, reward)
                self.save_done(env_id, done)
                success_times += 1

        # 处理结果
        valid_ids = [env_id for env_id in range(self.num_envs) if valid_envs[env_id]]
        if self.save_image:
            self.plot_vec()
        
        if len(valid_ids) == 0:
            for env in self.envs:
                ray.kill(env)
            return False
        else:
            for key in self.episode_buffer:
                self.episode_buffer[key] = torch.cat([self.episode_buffer[key][env_id, :self.env_pointers[env_id]] for env_id in valid_ids], dim=0)

            # 收集性能指标
            perf_metrices = []
            for env_id in range(self.num_envs):
                if valid_envs[env_id]:
                    env_id, perf_matrix = ray.get(self.envs[env_id]._get_perf_metrics.remote())
                    perf_metrices.append(perf_matrix)
            # 计算平均时间
            avg_predict_time = predict_time / step_count if step_count > 0 else 0
            avg_env_step_time = total_env_step_time / step_count if step_count > 0 else 0

            # 更新性能指标
            self.perf_metrics['travel_dist'] = np.sum([perf_matrix['travel_dist'] for perf_matrix in perf_metrices])
            self.perf_metrics['decision_steps'] = np.sum([self.env_pointers[env_id] for env_id in range(self.num_envs) if valid_envs[env_id]])
            self.perf_metrics['time_steps'] = np.sum([perf_matrix['time_steps'] for perf_matrix in perf_metrices])
            self.perf_metrics['max_time_steps'] = max([perf_matrix['time_steps'] for perf_matrix in perf_metrices])
            self.perf_metrics['max_num_nodes'] = max([perf_matrix['max_num_nodes'] for perf_matrix in perf_metrices])
            self.perf_metrics['max_num_vps'] = max([perf_matrix['max_num_vps'] for perf_matrix in perf_metrices])
            self.perf_metrics['max_k_size'] = max([perf_matrix['max_k_size'] for perf_matrix in perf_metrices])
            self.perf_metrics['episode_reward'] = np.sum([self.episode_buffer['rewards'].sum().item()])
            self.perf_metrics['success_rate'] = np.sum(success_times)
            self.perf_metrics['valid_envs'] = len(valid_ids)
            self.perf_metrics.update({
                # 时间统计
                'total_reset_time': reset_time,
                'total_predict_time': predict_time,
                'total_env_step_time': total_env_step_time,
                'avg_predict_time': avg_predict_time,
                'avg_env_step_time': avg_env_step_time,
                'step_count': step_count,
                'algo_type': self.algo_type  # 添加算法类型信息
            })

            for env in self.envs:
                ray.kill(env)
            return True
    
    def save_observation(self, env_id, observation):
        pointer = self.env_pointers[env_id]
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, viewpoints, viewpoint_padding_mask, adj_list = observation
        self.episode_buffer['node_inputs'][env_id, pointer] = node_inputs
        self.episode_buffer['node_padding_mask'][env_id, pointer] = node_padding_mask.bool()
        self.episode_buffer['current_index'][env_id, pointer] = current_index
        self.episode_buffer['adj_list'][env_id, pointer] = adj_list
        self.episode_buffer['viewpoints'][env_id, pointer] = viewpoints
        self.episode_buffer['viewpoint_padding_mask'][env_id, pointer] = viewpoint_padding_mask.bool()

    def save_action(self, env_id, action_index, logp):
        pointer = self.env_pointers[env_id]
        self.episode_buffer['actions'][env_id, pointer] = action_index.to(self.buffer_device)
        self.episode_buffer['logp'][env_id, pointer] = logp.to(self.buffer_device)

    def save_reward(self, env_id, reward):
        pointer = self.env_pointers[env_id]
        self.episode_buffer['rewards'][env_id, pointer] = torch.tensor([reward], dtype=torch.float32, device=self.buffer_device)

    def save_done(self, env_id, done):
        pointer = self.env_pointers[env_id]
        self.episode_buffer['dones'][env_id, pointer] = torch.tensor([done], dtype=torch.int8, device=self.buffer_device)
        self.env_pointers[env_id] += 1

    def get_perf_metrics(self):
        return self.perf_metrics
    
    def get_episode_buffer(self):
        return self.episode_buffer

    def plot_vec(self):
        for env in self.envs:
            ray.get(env._plot.remote())
            
    def close(self):
        for env in self.envs:
            ray.kill(env)

def create_ddql_worker(cfg, algo_cfg, meta_agent_id, **kwargs):
    """创建DDQL Worker"""
    return Worker.remote(cfg, algo_cfg, "ddql", meta_agent_id, **kwargs)


@hydra.main(config_path='./configs/small', config_name='config.yaml', version_base=None)
def test_ddql(cfg):
    """测试DDQL算法"""
    ray.init()
    seed = 2333
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    mp.set_start_method('spawn')
    cfg = preprocess_cfg(cfg) 

    save_img = True
    cfg_dict = OmegaConf.to_container(cfg)
    cfg_dict['runner']['num_envs_per_worker'] = 1
    cfg = OmegaConf.create(cfg_dict)
    
    time1 = time.time()
    algo_cfg = DDQLTrainConfig(
        device="cpu",
        seed=0,
        gnn_hidden_dim=cfg.graph.embedding_dim,
        viewpoint_padding_size=cfg.graph.viewpoint_padding_size,
        n_timesteps=10,
        num_inference_steps=10,  # 扩散推理步数
    )
    worker = create_ddql_worker(cfg=cfg, algo_cfg=algo_cfg, meta_agent_id=0, save_image=save_img)
    # checkpoint = torch.load('results/DDQL_replay_2_step_10_lr_3e4_gumbel_max_fixed_Q_fixed_V_fixed_entropy/models_1749785129/DDQL_replay_2_step_10_lr_3e4_gumbel_max_fixed_Q_fixed_V_fixed_entropy-pointer-e7fdf515/checkpoint_4000.pt', map_location='cpu', weights_only=False)
    checkpoint = torch.load('results/orig_ql_loss/models_1753032058/origin_ql_loss-57d3e15c/checkpoint_24000.pt', map_location='cpu', weights_only=False)
    ray.get(worker.init.remote(state_dict=checkpoint['actor']))
    # ray.get(worker.init.remote(state_dict=None))
    print('1. DDQL worker initialized')
    ray.get(worker.run_rollout.remote())
    print('2. perf_metrics:', ray.get(worker.get_perf_metrics.remote()))
    print('3. time cost:', time.time() - time1)
    worker.close.remote()
    ray.kill(worker)
    print('4. DDQL test completed')

@hydra.main(config_path='./configs/small', config_name='config.yaml', version_base=None)
def parallel_test_ddql(cfg):
    """DDQL算法的并行测试"""
    ray.init()
    seed = 2333
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    mp.set_start_method('spawn')
    cfg = preprocess_cfg(cfg)
    
    cfg_dict = OmegaConf.to_container(cfg)
    cfg_dict['runner']['num_envs_per_learner'] = 40
    cfg_dict['runner']['num_envs_per_worker'] = 1
    cfg = OmegaConf.create(cfg_dict)
    
    print(f"1. Creating DDQL workers")
    start_time = time.time()
    num_workers = cfg.runner.num_envs_per_learner // cfg.runner.num_envs_per_worker
    
    # 创建DDQL配置
    algo_cfg = DDQLTrainConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,
        gnn_hidden_dim=cfg.graph.embedding_dim,
        viewpoint_padding_size=cfg.graph.viewpoint_padding_size,
        n_timesteps=10,
    )
    
    workers = [create_ddql_worker(cfg, algo_cfg, i, save_image=False) for i in range(num_workers)]
    worker_creation_time = time.time() - start_time

    # 初始化各 worker
    init_start_time = time.time()
    # checkpoint = torch.load('results/DDQL_new_expert_T_10/models_1747645572/DDQL_new_expert_T_10-aaf33304/best_eval_model.pt', map_location='cpu', weights_only=False)
    # refs = [worker.init.remote(state_dict=checkpoint['actor']) for worker in workers]
    refs = [worker.init.remote(state_dict=None) for worker in workers]
    ray.get(refs)
    init_time = time.time() - init_start_time
    
    rollout_start_time = time.time()
    refs = [worker.run_rollout.remote() for worker in workers]
    valid_workers = ray.get(refs)
    rollout_time = time.time() - rollout_start_time
    
    # 合并rollout_data
    merge_start_time = time.time()
    rollout_data_list = [ray.get(worker.get_episode_buffer.remote()) for i, worker in enumerate(workers) if valid_workers[i]]
    perf_metrics_list = [ray.get(worker.get_perf_metrics.remote()) for i, worker in enumerate(workers) if valid_workers[i]]

    rollout_data = {}
    for key in rollout_data_list[0].keys():
        rollout_data[key] = torch.cat([data[key] for data in rollout_data_list], dim=0)
    merge_time = time.time() - merge_start_time
    
    # 合并perf_metrics
    metrics_start_time = time.time()
    perf_metrics = {
        'travel_dist': sum(metrics['travel_dist'] for metrics in perf_metrics_list),
        'success_rate': sum(metrics['success_rate'] for metrics in perf_metrics_list),
        'episode_reward': sum(metrics['episode_reward'] for metrics in perf_metrics_list),
        'valid_envs': sum(metrics['valid_envs'] for metrics in perf_metrics_list),
        'decision_steps': np.sum([metrics['decision_steps'] for metrics in perf_metrics_list]),
        'time_steps': np.sum([metrics['time_steps'] for metrics in perf_metrics_list]),
        'max_num_nodes': max([metrics['max_num_nodes'] for metrics in perf_metrics_list]),
        'max_num_vps': max([metrics['max_num_vps'] for metrics in perf_metrics_list]),
        'max_num_k_size': max([metrics['max_k_size'] for metrics in perf_metrics_list]),
        'max_time_steps': max([metrics['max_time_steps'] for metrics in perf_metrics_list]),
    }
    
    time_metrics = {
        'total_reset_time': sum(metrics['total_reset_time'] for metrics in perf_metrics_list),
        'total_predict_time': sum(metrics['total_predict_time'] for metrics in perf_metrics_list),
        'total_env_step_time': sum(metrics['total_env_step_time'] for metrics in perf_metrics_list),
        'max_env_step_time': max(metrics['total_env_step_time'] for metrics in perf_metrics_list),
        'step_count': sum(metrics['step_count'] for metrics in perf_metrics_list),
        'sampling_time': time.time() - start_time,
        'worker_creation_time': worker_creation_time,
        'initialization_time': init_time,
        'rollout_time': rollout_time,
        'buffer_merge_time': merge_time,
    }
    metrics_time = time.time() - metrics_start_time
    print(f"Metrics processing time: {metrics_time:.2f} seconds")
    
    perf_metrics.update({
        'avg_predict_time': time_metrics['total_predict_time'] / time_metrics['step_count'] if time_metrics['step_count'] > 0 else 0,
        'avg_env_step_time': time_metrics['total_env_step_time'] / time_metrics['step_count'] if time_metrics['step_count'] > 0 else 0,
        'max_env_step_time': time_metrics['max_env_step_time'],
    })
    
    travel_dist = torch.tensor(perf_metrics['travel_dist'], dtype=torch.float32)
    success_rate = torch.tensor(perf_metrics['success_rate'], dtype=torch.float32)
    episode_reward = torch.tensor(perf_metrics['episode_reward'], dtype=torch.float32)
    valid_envs = torch.tensor(perf_metrics['valid_envs'], dtype=torch.float32)
    rollout_size = torch.tensor(rollout_data['rewards'].size(0), dtype=torch.float32)
    decision_steps = torch.tensor(perf_metrics['decision_steps'], dtype=torch.float32)
    time_steps = torch.tensor(perf_metrics['time_steps'], dtype=torch.float32)
    sampling_time = torch.tensor(time_metrics['sampling_time'], dtype=torch.float32)

    travel_dist /= valid_envs
    success_rate /= valid_envs
    episode_reward /= valid_envs
    decision_steps /= valid_envs
    time_steps /= valid_envs

    print(f"Valid Env: {valid_envs.item()}, Valid Steps: {rollout_size.item()}, Sampling Time: {sampling_time.item()} s")

    print(
        f"DDQL Parallel Test Results:\n"
        f"reward={episode_reward.item():.3f}, travel_dist={travel_dist.item():.3f},\n"
        f"Performance: decision_steps={perf_metrics['decision_steps']:.3f}, time_steps={perf_metrics['time_steps']:.3f}, success_rate={success_rate.item():.3f}, max_num_nodes={perf_metrics['max_num_nodes']:.2f}, max_num_vps={perf_metrics['max_num_vps']:.2f}, max_num_k_size={perf_metrics['max_num_k_size']:.2f}, max_time_steps={perf_metrics['max_time_steps']:.2f}\n"
        f"Time Metrics: avg_predict_time={perf_metrics['avg_predict_time']:.3f}, avg_env_step_time={perf_metrics['avg_env_step_time']:.3f}, max_env_step_time={perf_metrics['max_env_step_time']}\n"
        )
    
    print("\nDDQL Time Statistics Summary:")
    print(f"Worker creation: {time_metrics['worker_creation_time']:.2f} s")
    print(f"Worker initialization: {time_metrics['initialization_time']:.2f} s")
    print(f"Rollout execution: {time_metrics['rollout_time']:.2f} s")
    print(f"Buffer merging: {time_metrics['buffer_merge_time']:.2f} s")
    print(f"Metrics processing: {metrics_time:.2f} s")
    print(f"Total time: {sampling_time.item():.2f} s")


if __name__ == "__main__":
    # 选择要测试的算法
    test_ddql()
    # test_iql()
    # parallel_test_iql()
    # parallel_test_ddql()
