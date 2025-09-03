# 神经网络TSP集成使用说明

## 概述

本功能将训练好的DDQL神经网络模型集成到EPIC探索系统中，用于替代传统的LKH TSP求解器进行智能视点选择。

## 配置方法

### 方法1：通过配置文件启用（推荐）

在相应的地图配置文件中（如`garage.yaml`, `cave.yaml`等）设置：

```yaml
# 神经网络TSP配置
neural_tsp/use_neural_tsp: true  # 启用神经网络TSP
neural_tsp/model_path: "/path/to/your/trained_model.pth"
neural_tsp/device: "cuda"  # 或 "cpu"
neural_tsp/node_dim: 6
neural_tsp/gnn_hidden_dim: 128
neural_tsp/max_viewpoints: 180
neural_tsp/n_timesteps: 100
neural_tsp/diffusion_mode: "pointer"  # 或 "simple"
```

然后正常启动：
```bash
roslaunch epic_planner garage.launch
```

### 方法2：通过启动参数覆盖

```bash
roslaunch epic_planner garage.launch use_neural_tsp:=true
```

## 模型要求

1. 模型应该是使用DDQL算法训练的
2. 输入格式需要兼容EPIC拓扑图数据结构：
   - 节点特征：`[x, y, z, yaw, is_viewpoint, is_current]` (6维)
   - 支持变长视点序列
   - 支持图结构输入

3. 输出：单个视点索引

## 实现细节

### 系统架构

```
C++探索管理器 <---ROS服务---> Python神经网络服务
     ↓                              ↓
原始TSP求解  ←--条件替换-->   DDQL模型推理
  (LKH)                        (PyTorch)
```

### 关键文件

- **C++端**：
  - `fast_exploration_manager.h/cpp`: 主要修改，添加神经网络TSP支持
  - `NeuralTSP.srv`: ROS服务定义

- **Python端**：
  - `neural_tsp_server.py`: 神经网络TSP推理服务

- **配置**：
  - `algorithm.xml`: 算法启动配置
  - `{map}.yaml`: 各地图配置文件

### 工作流程

1. 系统启动时根据配置决定是否启动神经网络服务
2. 探索过程中，当需要TSP求解时：
   - 如果启用神经网络且服务可用：调用神经网络推理
   - 否则：回退到传统LKH求解器
3. 返回选择的下一个视点索引

## 性能对比

- **传统LKH**：精确求解，但计算时间较长（数百毫秒）
- **神经网络**：近似求解，计算时间短（数十毫秒），需要训练

## 故障处理

1. **神经网络服务不可用**：自动回退到LKH求解器
2. **模型加载失败**：使用mock模型（随机选择）并发出警告
3. **推理失败**：回退到LKH求解器

## 调试

- 查看日志中的 "Neural TSP" 相关信息
- 检查服务状态：`rosservice list | grep neural_tsp`
- 测试服务：`rosservice call /neural_tsp_solve`

## 扩展指南

要集成你自己训练的模型，需要：

1. 修改 `neural_tsp_server.py` 中的模型加载部分
2. 确保模型输入输出格式与接口兼容
3. 根据需要调整观察值转换函数 `topo_graph_to_observation()`

## 编译

确保添加了消息生成依赖后重新编译：

```bash
cd /home/amax/EPIC
catkin build epic_planner
```

## 示例启动命令

```bash
# 使用传统TSP
roslaunch epic_planner garage.launch

# 使用神经网络TSP
roslaunch epic_planner garage.launch use_neural_tsp:=true

# 或者修改garage.yaml中的neural_tsp/use_neural_tsp为true
```
