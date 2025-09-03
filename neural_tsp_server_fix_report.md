# Neural TSP Server 数据处理一致性修复报告

## 问题描述

在检查`neural_tsp_server.py`和`epic3d_data_processor.py`的数据处理逻辑时，发现了以下不一致问题：

### 1. 邻接表构建不一致
- **epic3d_data_processor.py**: 只存储单向边 `from_node_id -> to_node_id`
- **neural_tsp_server.py**: 错误地添加了双向边（无向图）
- **影响**: 会导致图结构完全不同，影响GNN的图卷积计算

### 2. 节点特征构建不一致  
- **特征顺序不同**: 两者使用了不同的特征向量排列
- **位置表示不同**: epic3d使用相对位置，neural_tsp使用绝对位置
- **特征计算方式不同**: 标准化和处理逻辑存在差异
- **影响**: 会导致模型输入完全错误，无法正确推理

### 3. 视点索引映射逻辑不一致
- **construct_observation**: 使用节点索引构建viewpoint_indices
- **handle_tsp_request**: 使用错误的映射逻辑解析模型输出
- **影响**: 导致选择的视点索引映射错误

## 修复方案

### 1. 统一邻接表构建逻辑

```python
# 修复前（错误）
adjacency_dict[from_node].append(to_node)
adjacency_dict[to_node].append(from_node)  # 错误：添加双向边

# 修复后（正确）  
adjacency_dict[from_node].append(to_node)  # 只添加单向边
# 不添加反向边，与epic3d_data_processor.py保持一致
```

### 2. 统一节点特征构建逻辑

```python
# 修复后的特征向量顺序（与epic3d_data_processor.py完全一致）
feature_vector = [
    rel_pos[0], rel_pos[1], rel_pos[2],      # 相对3D位置（不是绝对位置）
    obs_score,                                # 标准化观测得分
    cluster_dist,                            # 标准化聚类距离
    is_viewpoint,                            # 节点类型
    is_history,                              # 访问状态（不是is_current）
    distance,                                # 距离特征
    0.5                                      # 占位的中心性得分
]
```

### 3. 修正视点索引映射逻辑

```python
# 修复后的映射逻辑
# 1. 构建viewpoint_nodes列表（与construct_observation中的逻辑一致）
viewpoint_nodes = []
for i in range(min(len(req.node_positions), len(req.node_is_viewpoint))):
    if req.node_is_viewpoint[i]:
        viewpoint_nodes.append(i)

# 2. 模型输出是在viewpoint_nodes中的索引
selected_node_idx = action_indices[0, 0].cpu().numpy()

# 3. 获取实际的拓扑节点索引
if selected_node_idx < len(viewpoint_nodes):
    selected_topo_node = viewpoint_nodes[selected_node_idx]
    # 然后映射到req.viewpoint_indices中的位置
```

## 修复验证

### 测试脚本验证
创建了`test_data_consistency.py`测试脚本，验证：
- ✅ 邻接表构建逻辑一致性
- ✅ 节点特征构建逻辑一致性  
- ✅ 数值计算精度一致性

### 测试结果
```
=== 测试邻接表构建一致性 ===
✅ 邻接表构建逻辑一致

=== 测试节点特征构建一致性 ===  
✅ 节点特征构建逻辑一致

✅ 所有测试通过！数据处理逻辑一致。
```

## NeuralTSP.srv 更新

基于数据流水线文档的分析，更新了服务消息定义，添加了construct_observation所需的完整拓扑图信息：

```protobuf
# 拓扑图节点信息
geometry_msgs/Point[] node_positions    # 所有节点的3D位置
float32[] node_yaws                     # 节点航向角
bool[] node_is_viewpoint                # 节点是否为视点
bool[] node_is_current                  # 节点是否为当前位置  
bool[] node_is_history                  # 节点是否为历史位置
int32[] node_region_ids                 # 节点所属区域ID
bool[] node_is_reachable               # 节点是否可达
int32[] node_tsp_order                 # 节点在TSP中的顺序(-1表示无效)
float32[] node_distances               # 节点到当前位置的距离
float32[] node_observation_scores      # 节点观测收益评分
float32[] node_cluster_distances       # 节点到聚类中心的距离

# 拓扑图边信息
int32[] edge_from_nodes                # 边的起始节点ID
int32[] edge_to_nodes                  # 边的终止节点ID  
float32[] edge_weights                 # 边的权重
bool[] edge_is_reachable              # 边是否可达
```

## 关键修复点

1. **数据格式完全对齐**: neural_tsp_server现在与epic3d_data_processor使用完全相同的数据处理逻辑
2. **图结构一致性**: 邻接表构建逻辑保持一致（单向边）
3. **特征工程一致性**: 节点特征构建使用相同的归一化和特征组合方式
4. **索引映射正确性**: 修正了模型输出到实际视点的映射逻辑

## 预期效果

修复后的neural_tsp_server应该能够：
- 正确处理ROS服务请求中的拓扑图数据
- 构建与训练时完全一致的模型输入
- 准确解析模型输出并映射到正确的视点选择
- 提供可靠的神经网络TSP求解服务

## 下一步计划

1. 在实际EPIC仿真环境中测试neural_tsp_server
2. 验证端到端的神经TSP决策流程  
3. 对比神经TSP与传统LKH的性能差异
4. 根据测试结果进一步优化服务性能
