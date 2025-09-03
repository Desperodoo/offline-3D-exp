# Neural TSP配置指南

## 配置架构

为了简化配置管理，Neural TSP的配置已统一到`algorithm.xml`中作为唯一配置入口。

### 配置流程

```
环境launch文件 (e.g., cave.launch)
    ↓ 传递参数 use_neural_tsp
algorithm.xml 
    ↓ 设置ROS参数 neural_tsp/use_neural_tsp
FastExplorationManager
    ↓ 读取并使用配置
```

## 使用方法

### 方法1：修改默认值（推荐）
直接在`algorithm.xml`中修改默认值：
```xml
<arg name="use_neural_tsp" default="true" />  <!-- 改为true -->
```

### 方法2：启动时指定参数
```bash
# 启用neural TSP
roslaunch epic_planner cave.launch use_neural_tsp:=true

# 禁用neural TSP (默认)
roslaunch epic_planner cave.launch use_neural_tsp:=false
```

### 方法3：使用专门的neural_tsp.launch
```bash
roslaunch epic_planner neural_tsp.launch
```

## 配置文件说明

- `algorithm.xml` - **唯一配置入口**，包含neural_tsp的launch参数定义
- `*.launch` - 各环境的启动文件，只传递参数不定义配置
- `*.yaml` - 配置文件中**不再**包含neural_tsp相关配置

## 重要说明

1. **所有yaml配置文件都不包含neural_tsp配置** - 统一由launch参数控制
2. **algorithm.xml是所有环境通用的** - 确保配置一致性
3. **neural_tsp_server会自动根据条件启动** - 在algorithm.xml中定义

## 故障排除

如果neural_tsp不工作，检查：
1. `use_neural_tsp`参数是否正确传递到algorithm.xml
2. neural_tsp_server是否正确启动
3. 查看launch输出中的neural TSP相关日志
