# Enhanced Topology Extraction System

## 功能升级总结

我们已经成功改进了 `topo_graph_integrated_extractor`，新增了以下关键功能：

### 1. 新增的NodeInfo字段
```cpp
struct NodeInfo {
    int node_id;
    Eigen::Vector3d position;
    double yaw;
    bool is_viewpoint;
    bool is_current_odom;
    bool is_history_odom;
    int region_id;
    bool is_reachable;          // 新增：节点是否可达（从viewpoint过滤中获取）
    int tsp_order_index;        // 新增：在TSP求解结果中的顺序索引 (-1表示未参与TSP)
    double reachable_distance;  // 新增：到该节点的可达距离
};
```

### 2. viewpoint reachability特征
- **数据来源**：从 `planGlobalPath` 第4步的视点过滤结果中获取
- **判断逻辑**：如果viewpoint在`viewpoint_reachable`列表中，则`is_reachable = true`
- **距离信息**：同时记录`reachable_distance`，表示从当前位置到该viewpoint的可达距离

### 3. TSP求解结果的整合
- **TSP顺序索引**：`tsp_order_index`字段记录每个viewpoint在TSP求解结果中的访问顺序
- **索引含义**：
  - `0`：起始位置（通常是current odom节点）
  - `1, 2, 3...`：TSP求解确定的viewpoint访问顺序
  - `-1`：未参与TSP求解的节点

### 4. 增强版提取流程
在`FastExplorationManager::planGlobalPath`中：
```cpp
// TSP求解完成后，更新viewpoint信息
if (topo_extractor_) {
    ROS_INFO("Updating viewpoint info: %zu reachable, TSP order size: %zu", 
             viewpoint_reachable.size(), indices.size());
    topo_extractor_->updateViewpointInfo(viewpoint_reachable, 
                                        viewpoint_reachable_distance2, 
                                        indices);
    
    // 重新提取拓扑图以包含更新的viewpoint信息
    topo_extractor_->extractTopoGraph("planGlobalPath_final");
}
```

### 5. 输出格式示例
新的导出文件格式：
```
# 节点格式: node_id x y z yaw is_viewpoint is_current is_history region_id is_reachable tsp_order distance
NODES
0 10.5 20.3 1.5 1.57 1 0 0 100 1 2 15.3
1 5.2 18.7 1.5 0.78 1 0 0 101 1 1 8.9
2 15.8 25.1 1.5 2.34 1 0 0 102 1 3 22.1
3 0.0 0.0 1.5 0.0 0 1 0 -1 1 0 0.0
...
```

各字段含义：
- `is_reachable`: 1表示可达，0表示不可达
- `tsp_order`: TSP中的访问顺序（0=起点，-1=未参与TSP）
- `distance`: 可达距离（-1.0表示不可达或未计算）

### 6. 关键改进点

#### 数据一致性
- 利用planGlobalPath中已经计算好的reachability判断
- 直接使用TSP求解器的输出结果
- 避免重复计算，提高效率

#### 时序同步
- 在TSP求解完成后立即更新viewpoint信息
- 确保提取的数据反映最新的规划状态

#### 调试支持
- 增强版日志输出，清楚显示reachability和TSP信息
- 通过ROS参数控制调试输出级别

### 7. 使用示例

启用调试输出：
```bash
rosparam set /exploration_node/topo_extraction/debug_output true
```

查看增强信息：
```bash
# 查看提取的文件
ls /tmp/topo_graph_*.txt

# 查看最新提取结果
tail -50 /tmp/topo_graph_$(ls -1t /tmp/topo_graph_*.txt | head -1 | sed 's|.*/topo_graph_||' | sed 's|\.txt||').txt
```

这个增强系统现在能够：
1. ✅ 准确记录viewpoint的可达性状态
2. ✅ 保存TSP求解的访问顺序
3. ✅ 提供距离信息便于分析
4. ✅ 与现有EPIC系统无缝集成

系统已经成功编译，可以在实际探索场景中进行测试验证。
