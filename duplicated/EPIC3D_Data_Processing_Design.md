# EPIC 3D数据处理模块设计方案

## 概述

本设计方案旨在将EPIC 3D探索系统收集的数据转换为适用于离线强化学习训练的格式，无需修改现有的C++数据采集Pipeline。

## 数据流分析

### 现有EPIC 3D数据格式
```
collected_data/
├── batch_name/
│   ├── config.txt                    # 运行配置
│   ├── filtered_data/               # 过滤后的时序数据
│   │   ├── topo_graph_timestamp1.txt
│   │   ├── topo_graph_timestamp2.txt
│   │   └── ... (N个时间步文件)
│   ├── raw_data/                    # 原始数据
│   └── roslaunch.log
```

### 数据格式结构
每个topo_graph文件包含:
- **探索统计信息**: total_distance, exploration_area, viewpoints_visited等
- **节点数据**: node_id x y z yaw is_viewpoint is_current is_history region_id is_reachable tsp_order_index distance observation_score cluster_distance
- **边数据**: edge_id from_node_id to_node_id weight is_reachable

## 数据处理Pipeline设计

### Phase 1: EPIC3D数据加载器 (`epic3d_data_processor.py`)

#### 1.1 数据结构定义
```python
EPIC3DTimeStep = namedtuple('EPIC3DTimeStep', [
    'timestamp',          # 时间戳
    'exploration_stats',  # 探索统计信息
    'nodes',             # 节点信息列表
    'edges',             # 边信息列表
    'viewpoints',        # 视点节点信息
    'current_pos',       # 当前位置
    'trajectory_info'    # 轨迹信息
])
```

#### 1.2 核心组件
- **EPIC3DDataParser**: 解析单个topo_graph文件
- **EPIC3DEpisodeProcessor**: 处理单个episode的所有时间步
- **EPIC3DDatasetBuilder**: 构建完整的离线RL数据集

#### 1.3 特征工程设计
节点特征向量 (9维):
```
[rel_x, rel_y, rel_z, obs_score, cluster_dist, is_vp, visited, dist, centrality]
```
- `rel_x, rel_y, rel_z`: 相对于当前位置的3D坐标 (标准化)
- `obs_score`: 观测得分 (标准化到0-1)
- `cluster_dist`: 集群距离成本 (标准化)
- `is_vp`: 是否为视点 (二值)
- `visited`: 是否已访问 (二值)
- `dist`: 到当前位置距离 (标准化)
- `centrality`: 中心性得分 (占位，固定值0.5)

### Phase 2: 数据集构建脚本 (`build_epic3d_dataset.py`)

#### 2.1 主要功能
- 批量处理多个episode目录
- 支持命令行参数配置
- 生成HDF5格式的训练数据集
- 提供详细的处理日志

#### 2.2 输出格式
与2D项目兼容的HDF5数据集:
```python
{
    'node_inputs': (T, max_nodes, node_dim),
    'node_padding_mask': (T, 1, max_nodes),
    'current_index': (T, 1, 1),
    'viewpoints': (T, max_viewpoints, 1),
    'viewpoint_padding_mask': (T, 1, max_viewpoints),
    'adj_list': (T, max_nodes, k_size),
    'actions': (T,),
    'rewards': (T,),
    'dones': (T,)
}
```

### Phase 3: 配置文件 (`epic3d_rl_config.yaml`)

#### 3.1 数据处理配置
```yaml
data_processing:
  max_nodes: 500          # 最大节点数
  max_viewpoints: 100     # 最大视点数
  k_size: 20             # 邻接表K近邻
  node_feature_dim: 9    # 节点特征维度
  
  # 标准化参数
  position_scale: 100.0
  observation_score_scale: 50.0
  cluster_distance_scale: 20.0
  distance_scale: 50.0
```

## 动作和奖励设计

### 动作推断策略
EPIC数据中包含TSP顺序信息，可以直接用于动作推断:
1. 使用`tsp_order_index`字段获取动作信息
2. `tsp_order_index = 0`: 当前位置
3. `tsp_order_index = 1`: 下一个目标点 (即选择的动作)
4. 通过查找`tsp_order_index = 1`的视点节点确定动作标签

### 奖励函数设计
```python
reward = -(current_total_distance - previous_total_distance)
```
- **移动奖励**: 基于前后两帧`total_distance`的差值
- **负号**: 距离增加给予负奖励，鼓励高效路径
- **简化设计**: 不考虑探索覆盖率，专注于路径效率

## 关键特性

### 优势
1. **无需修改现有Pipeline**: 完全基于现有topo_graph文件
2. **兼容2D训练架构**: 输出格式与现有graph_buffer兼容
3. **批量处理能力**: 支持多episode并行处理
4. **高效存储**: HDF5格式支持压缩和快速访问
5. **灵活配置**: 易于调整特征工程和参数

### 数据质量保证
1. **数据验证**: 检查节点和边的一致性
2. **异常处理**: 跳过损坏的文件并记录日志
3. **标准化**: 统一的特征标准化策略
4. **元数据保存**: 完整保留episode统计信息

## 使用流程

### 步骤1: 准备配置文件
```bash
# 编辑配置文件设置处理参数
vim epic3d_rl_config.yaml
```

### 步骤2: 构建数据集
```bash
python build_epic3d_dataset.py \
    --data_dirs /home/amax/EPIC/collected_data \
    --output /home/amax/EPIC/datasets/epic3d_offline_dataset.h5 \
    --config epic3d_rl_config.yaml
```

### 步骤3: 验证数据集
```bash
python validate_dataset.py \
    --dataset /home/amax/EPIC/datasets/epic3d_offline_dataset.h5
```

## 扩展性设计

### 特征扩展
- 可以轻松添加新的节点特征维度
- 支持不同的标准化策略
- 可以集成更复杂的图结构特征

### 动作空间扩展
- 支持连续动作空间
- 可以添加多步动作预测
- 支持层次化动作结构

### 奖励函数扩展
- 可以添加更复杂的探索奖励
- 支持多目标奖励函数
- 可以集成安全性约束

## 技术细节

### 内存优化
- 使用numpy数组进行高效数值计算
- HDF5压缩减少存储空间
- 流式处理避免内存溢出

### 并行处理
- 支持多进程episode处理
- 可以分布式处理大规模数据集
- 支持增量数据集构建

### 错误处理
- 完善的异常捕获和日志记录
- 数据完整性检查
- 自动跳过损坏文件

这个设计方案充分利用了EPIC现有的丰富数据，通过合理的特征工程和数据转换，可以直接用于离线强化学习训练，为EPIC 3D探索系统提供学习能力。
