# EPIC神经网络TSP集成指南

## 概述

本指南介绍如何将训练好的神经网络模型集成到EPIC项目中，替代传统的LKH TSP求解器。

## 系统架构

```
C++ 探索管理器 (FastExplorationManager)
    ↓ ROS服务调用
Python TSP服务 (neural_tsp_server.py) 
    ↓ 模型推理
DDQL神经网络模型
    ↓ 输出
下一个目标视点索引
```

## 文件结构

```
src/global_planner/exploration_manager/
├── srv/
│   └── NeuralTSP.srv                    # ROS服务定义
├── scripts/
│   └── neural_tsp_server.py            # Python推理服务
├── launch/
│   ├── neural_tsp.launch               # 神经网络服务启动文件
│   └── garage_neural.launch            # 带神经网络的主启动文件
├── include/epic_planner/
│   └── fast_exploration_manager.h      # 头文件（已修改）
└── src/
    └── fast_exploration_manager.cpp    # 主要实现（已修改）
```

## 使用方法

### 1. 编译系统

```bash
cd /home/amax/EPIC
catkin build epic_planner
source devel/setup.bash
```

### 2. 准备模型文件

将训练好的DDQL模型文件放置到合适位置，例如：
```bash
mkdir -p src/global_planner/exploration_manager/models
# 将你的模型文件复制到这里，例如：ddql_model.pth
```

### 3. 配置神经网络服务

编辑 `launch/neural_tsp.launch`，修改以下参数：
- `model_path`: 模型文件路径
- `node_dim`: 节点特征维度
- `gnn_hidden_dim`: GNN隐藏层维度
- `max_viewpoints`: 最大视点数量
- `diffusion_mode`: 扩散模式（"pointer" 或 "simple"）

### 4. 启动系统

#### 方式1：使用神经网络TSP（推荐用于测试）
```bash
roslaunch epic_planner garage_neural.launch use_neural_tsp:=true
```

#### 方式2：使用传统LKH求解器（默认）
```bash
roslaunch epic_planner garage.launch
```

#### 方式3：运行时切换
```bash
# 启动系统（默认使用LKH）
roslaunch epic_planner garage.launch

# 在另一个终端启动神经网络服务
roslaunch epic_planner neural_tsp.launch

# 动态启用神经网络TSP
rosparam set /exploration_node/exploration/use_neural_tsp true
```

## 配置参数

### C++端参数（在launch文件中设置）
- `exploration/use_neural_tsp`: 是否启用神经网络TSP（默认false）
- `exploration/tsp_dir`: 传统LKH求解器目录

### Python端参数
- `model_path`: 模型文件路径
- `device`: 运行设备（"cuda" 或 "cpu"）
- `node_dim`: 节点特征维度（默认6）
- `gnn_hidden_dim`: GNN隐藏维度（默认128）
- `max_viewpoints`: 最大视点数（默认180）

## 集成你的DDQL模型

目前`neural_tsp_server.py`使用的是模拟模型。要集成你的真实模型，请修改`load_model()`方法：

```python
def load_model(self):
    """加载训练好的DDQL模型"""
    try:
        # 添加你的offline模块路径
        sys.path.append('/path/to/your/offline/directory')
        from agent.ddql import initialize_ddql, TrainConfig
        
        # 创建配置
        config = TrainConfig()
        config.node_dim = rospy.get_param('~node_dim', 6)
        config.gnn_hidden_dim = rospy.get_param('~gnn_hidden_dim', 128)
        config.max_viewpoints = rospy.get_param('~max_viewpoints', 180)
        config.n_timesteps = rospy.get_param('~n_timesteps', 100)
        config.diffusion_mode = rospy.get_param('~diffusion_mode', 'pointer')
        config.device = str(self.device)
        config.load_model_path = rospy.get_param('~model_path', '')
        
        # 初始化并加载模型
        self.model = initialize_ddql(config, rank=0, world_size=1)
        rospy.loginfo(f"Real DDQL model loaded from {config.load_model_path}")
        
    except Exception as e:
        rospy.logerr(f"Failed to load DDQL model: {e}")
        self.model = MockDDQLModel(self.device)
```

## 性能对比

系统会在日志中输出TSP求解时间和使用的求解器类型：
```
calculate tsp cost matrix cost X ms
tsp solver cost: Y ms  
solver type: Neural  # 或 LKH
```

## 故障排除

### 1. 服务连接失败
- 检查`neural_tsp_server.py`是否正确启动
- 确认ROS服务`/neural_tsp_solve`是否存在：`rosservice list | grep neural`

### 2. 模型加载失败
- 检查模型文件路径是否正确
- 确认Python环境中安装了PyTorch和相关依赖
- 查看`neural_tsp_server`节点的日志输出

### 3. 推理结果异常
- 检查输入数据格式是否与训练时一致
- 确认模型配置参数正确
- 可以先用模拟模型测试整个流程

### 4. 编译错误
- 确保安装了`message_generation`包：`sudo apt install ros-noetic-message-generation`
- 重新编译：`catkin build epic_planner --force-cmake`

## 扩展功能

### 添加更多输入信息
如需要传递更多拓扑图信息给神经网络，可以修改：
1. `NeuralTSP.srv`服务定义
2. `solveNeuralTSP()`函数中的请求构建部分
3. `handle_tsp_request()`函数中的数据处理部分

### 批量推理优化
对于多视点场景，可以考虑批量推理以提高效率：
```python
# 在handle_tsp_request中实现批量处理
actions = self.model.select_actions_batch(observations)
```

## 注意事项

1. **回退机制**：系统具备自动回退功能，神经网络求解失败时会自动使用LKH求解器
2. **实时性要求**：神经网络推理时间应控制在100ms内以保证实时性
3. **内存管理**：长时间运行时注意Python进程的内存使用情况
4. **模型兼容性**：确保模型输入输出格式与接口一致
