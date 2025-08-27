# EPIC 数据自动采集系统使用说明

## 概述

这个系统基于您现有的EPIC项目，实现了完全自动化的数据采集流程，包括：
- 随机地图和初始位置选择
- launch文件自动修改
- ROS系统启动和数据采集
- 自动进程终止和数据过滤

## 文件结构

```
/home/amax/EPIC/
├── data_collection_script.py      # 主要采集脚本
├── run_data_collection.sh         # 便捷启动脚本
├── filter_topo_v2.py              # 数据过滤脚本（您已有的）
├── collected_data/                 # 采集数据输出目录
└── src/MARSIM/map_generator/resource/
    ├── forest.pcd                  # 地图文件
    ├── partition.pcd
    ├── dungeon.pcd
    └── forest/batch_*/             # 初始位置配置
        ├── 0_map_free_points.txt   # 50个初始位置坐标
        ├── 1_map_free_points.txt
        └── ...
```

## 可用地图和配置

系统当前支持30个配置组合：
- **forest**: 10个批次 (batch_1 到 batch_10)，每个批次包含10个不同的初始位置文件
- **partition**: 10个批次，每个批次包含10个不同的初始位置文件  
- **dungeon**: 10个批次，每个批次包含10个不同的初始位置文件

每个初始位置文件包含50个经过筛选的安全起始点，距离障碍物至少2米。

## 使用方法

### 1. 基本用法

```bash
# 进入项目目录
cd /home/amax/EPIC

# 单次随机采集 (5分钟)
python3 data_collection_script.py -t 300

# 快速测试 (1分钟)
python3 data_collection_script.py -t 60

# 批量采集 (3次，每次5分钟)
python3 data_collection_script.py -c 3 -t 300
```

### 2. 指定参数

```bash
# 指定地图类型
python3 data_collection_script.py --map-type forest -t 300

# 指定感知范围
python3 data_collection_script.py -s 20.0 -t 300

# 长时间采集 (10分钟)
python3 data_collection_script.py -t 600
```

### 3. 使用便捷脚本

```bash
# 交互式菜单
./run_data_collection.sh

# 或直接传参
./run_data_collection.sh -c 5 -t 300
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-c, --count` | 采集次数 | 1 |
| `-t, --time` | 单次采集时长(秒) | 300 |
| `-d, --dir` | EPIC项目目录 | /home/amax/EPIC |
| `-s, --sensing-horizon` | 感知范围 | 15.0 |
| `--map-type` | 指定地图类型 | 随机选择 |
| `--list-configs` | 列出所有可用配置 | - |

## 输出结构

每次采集会在 `collected_data/` 下创建独立目录：

```
collected_data/
└── forest_batch_3_2_20250825_143022/
    ├── config.txt              # 采集配置信息
    ├── raw_data/               # 原始数据
    │   ├── topo_*.txt          # 拓扑图文件
    │   └── exploration_stats_*.txt  # 探索统计文件
    └── filtered_data/          # 过滤后的数据
        └── *.txt               # 过滤后的拓扑图文件
```

### 配置文件内容示例

```
# 数据采集配置 - 20250825_143022
map_type: forest
batch_id: batch_3
point_id: 2
init_position: 34.200000 26.200000 1.000000
init_yaw: 0.000000
pcd_file: /home/amax/EPIC/src/MARSIM/map_generator/resource/forest.pcd
txt_file: /home/amax/EPIC/src/MARSIM/map_generator/resource/forest/batch_3/2_map_free_points.txt
collection_time: 300
```

## 数据过滤规则

系统自动使用 `filter_topo_v2.py` 进行数据过滤：
1. 移除 `total_distance = 0` 的无效数据点
2. 过滤时间间隔小于0.5秒的重复数据
3. 保留探索起始点和有效的探索轨迹点

## 工作流程

1. **配置选择**: 从30个可用配置中随机选择一个地图和初始位置组合
2. **位置选择**: 从选中的初始位置文件中随机选择一个起始点坐标
3. **Launch修改**: 自动修改对应的launch文件，设置正确的地图和初始位置
4. **ROS启动**: 启动roscore和launch文件
5. **数据采集**: 等待指定时间，系统自动记录拓扑图和探索统计数据
6. **进程终止**: 自动终止所有ROS进程
7. **数据整理**: 移动原始数据到输出目录
8. **数据过滤**: 使用filter_topo_v2规则过滤数据
9. **清理**: 删除临时文件

## 注意事项

1. **ROS环境**: 确保ROS环境正确配置，项目已编译
2. **权限**: 脚本需要kill进程的权限
3. **磁盘空间**: 确保有足够的磁盘空间存储数据
4. **进程清理**: 脚本会自动清理ROS进程，但建议采集前检查无残留进程

## 故障排除

### 常见问题

1. **"Launch文件不存在"**
   - 检查launch文件路径是否正确
   - 确认对应地图的launch文件存在

2. **"未找到可用的配置文件"**
   - 检查resource目录结构是否完整
   - 确认txt文件格式正确

3. **ROS进程无法启动**
   - 检查ROS环境变量
   - 确认项目已正确编译
   - 检查是否有端口冲突

4. **数据过滤失败**
   - 检查filter_topo_v2.py是否存在且可执行
   - 确认Python环境正确

### 手动清理

如果脚本异常退出，可以手动清理：

```bash
# 杀死ROS进程
pkill -f roslaunch
pkill -f roscore
pkill -f epic_planner

# 检查残留进程
ps aux | grep ros
```

## 批量采集建议

对于大规模数据采集：
- 建议单次采集时间300-600秒
- 采集间隔至少10秒，让系统稳定
- 监控磁盘空间，及时清理旧数据
- 考虑在不同时段进行采集，避免系统负载过高

## 扩展功能

可以根据需要扩展的功能：
- 支持更多地图类型
- 自定义数据过滤规则
- 添加实时监控和状态报告
- 集成数据分析和可视化
- 支持分布式采集
