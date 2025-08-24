/**
 * @file topo_graph_integrated_extractor.h
 * @brief 集成的拓扑图提取器，用于从TopoGraph中提取节点和边信息
 */

#ifndef TOPO_GRAPH_INTEGRATED_EXTRACTOR_H
#define TOPO_GRAPH_INTEGRATED_EXTRACTOR_H

#include <pointcloud_topo/graph.h>
#include <ros/ros.h>
#include <std_srvs/Trigger.h>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <unordered_map>
#include <epic_planner/exploration_statistics.h>

namespace fast_planner {

// 节点信息结构
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

// 边信息结构
struct EdgeInfo {
    int edge_id;
    int from_node_id;
    int to_node_id;
    double weight;
    bool is_reachable;
};

// 拓扑图提取功能类
class TopoGraphExtractorInline {
public:
    /**
     * @brief 从TopoGraph中提取完整的节点和边信息
     * @param topo_graph 拓扑图指针
     * @param nodes 输出的节点信息列表
     * @param edges 输出的边信息列表
     */
    static void exportCompleteGraph(TopoGraph::Ptr topo_graph, 
                                   std::vector<NodeInfo>& nodes, 
                                   std::vector<EdgeInfo>& edges);
    
    /**
     * @brief 生成边连接信息
     * @param nodes 节点信息列表
     * @param edges 输出的边信息列表
     * @param edge_counter 边计数器
     * @param distance_threshold 距离阈值
     */
    static void generateEdges(const std::vector<NodeInfo>& nodes, 
                             std::vector<EdgeInfo>& edges, 
                             int& edge_counter,
                             double distance_threshold = 10.0);

private:
    /**
     * @brief 提取区域节点
     * @param topo_graph 拓扑图指针
     * @param nodes 节点信息列表
     * @param node_counter 节点计数器引用
     */
    static void extractRegionNodes(TopoGraph::Ptr topo_graph, 
                                  std::vector<NodeInfo>& nodes, 
                                  int& node_counter);

    /**
     * @brief 提取当前位置节点
     * @param topo_graph 拓扑图指针
     * @param nodes 节点信息列表
     * @param node_counter 节点计数器引用
     */
    static void extractOdomNode(TopoGraph::Ptr topo_graph, 
                               std::vector<NodeInfo>& nodes, 
                               int& node_counter);

    /**
     * @brief 提取视点节点
     * @param topo_graph 拓扑图指针
     * @param nodes 节点信息列表
     * @param node_counter 节点计数器引用
     */
    static void extractViewpointNodes(TopoGraph::Ptr topo_graph, 
                                     std::vector<NodeInfo>& nodes, 
                                     int& node_counter);
};

// 集成的拓扑图提取器类
class TopoExtractorIntegrated {
private:
    TopoGraph::Ptr topo_graph_;
    ExplorationStatistics::Ptr exploration_stats_;  // 新增：探索统计模块
    
    // 参数
    bool enable_extraction_;
    std::string export_dir_;
    double distance_threshold_;
    bool debug_output_;  // 是否输出详细调试信息
    
    // 可选的定时器功能（如果需要定时提取）
    ros::Timer extract_timer_;
    ros::ServiceServer extract_service_;
    ros::NodeHandle* nh_ptr_;
    
public:
    /**
     * @brief 构造函数
     * @param graph 拓扑图指针
     * @param export_dir 导出目录
     * @param enable_extraction 是否启用提取功能
     * @param distance_threshold 距离阈值
     */
    TopoExtractorIntegrated(TopoGraph::Ptr graph, 
                           const std::string& export_dir = "/tmp",
                           bool enable_extraction = true,
                           double distance_threshold = 10.0);

    /**
     * @brief 构造函数（带ROS节点句柄，用于定时器和服务）
     * @param nh ROS节点句柄
     * @param graph 拓扑图指针
     * @param enable_timer 是否启用定时器
     * @param extraction_rate 提取频率
     */
    TopoExtractorIntegrated(ros::NodeHandle& nh, TopoGraph::Ptr graph,
                           bool enable_timer = false, double extraction_rate = 0.1);

    /**
     * @brief 析构函数
     */
    ~TopoExtractorIntegrated() = default;

    /**
     * @brief 直接执行拓扑图提取（主要接口）
     * @param context 提取上下文信息，用于日志
     * @return 提取是否成功
     */
    bool extractTopoGraph(const std::string& context = "");
    
    /**
     * @brief 设置导出目录
     * @param dir 导出目录路径
     */
    void setExportDir(const std::string& dir) { export_dir_ = dir; }
    
    /**
     * @brief 设置距离阈值
     * @param threshold 距离阈值
     */
    void setDistanceThreshold(double threshold) { distance_threshold_ = threshold; }
    
    /**
     * @brief 启用或禁用提取功能
     * @param enable 是否启用
     */
    void setEnabled(bool enable) { enable_extraction_ = enable; }
    
    /**
     * @brief 设置调试输出
     * @param debug 是否启用调试输出
     */
    void setDebugOutput(bool debug) { debug_output_ = debug; }
    
    /**
     * @brief 更新viewpoint的reachability和TSP信息
     * @param reachable_viewpoints 可达的viewpoint节点列表
     * @param reachable_distances 到每个viewpoint的距离
     * @param tsp_indices TSP求解结果中的顺序索引 (从planGlobalPath获取)
     */
    void updateViewpointInfo(const std::vector<TopoNode::Ptr>& reachable_viewpoints,
                            const std::vector<double>& reachable_distances,
                            const std::vector<int>& tsp_indices);
    
    /**
     * @brief 设置探索统计模块
     * @param stats 探索统计模块指针
     */
    void setExplorationStats(ExplorationStatistics::Ptr stats) { exploration_stats_ = stats; }

private:
    // 存储viewpoint的扩展信息
    std::unordered_map<TopoNode::Ptr, bool> viewpoint_reachability_map_;
    std::unordered_map<TopoNode::Ptr, double> viewpoint_distance_map_;  
    std::unordered_map<TopoNode::Ptr, int> viewpoint_tsp_index_map_;

private:
    /**
     * @brief 定时器回调函数，定期提取拓扑图
     * @param event 定时器事件
     */
    void extractCallback(const ros::TimerEvent& event);
    
    /**
     * @brief ROS服务回调函数，手动触发提取
     * @param req 服务请求
     * @param res 服务响应
     * @return 服务处理结果
     */
    bool extractService(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);
    
    /**
     * @brief 将提取的数据导出为简单格式（包含探索统计信息）
     * @param nodes 节点信息列表
     * @param edges 边信息列表
     * @param filename 输出文件名
     */
    void exportSimpleFormat(const std::vector<NodeInfo>& nodes,
                           const std::vector<EdgeInfo>& edges,
                           const std::string& filename);
                           
    /**
     * @brief 初始化参数（从构造函数中提取的公共逻辑）
     * @param nh ROS节点句柄指针（可选）
     */
    void initializeParams(ros::NodeHandle* nh = nullptr);
    
    /**
     * @brief 增强版本的拓扑图提取（包含reachability和TSP信息）
     * @param nodes 输出的节点信息列表
     * @param edges 输出的边信息列表
     */
    void extractCompleteGraphEnhanced(std::vector<NodeInfo>& nodes, 
                                     std::vector<EdgeInfo>& edges);
    
    /**
     * @brief 增强版本的区域节点提取
     * @param nodes 输出的节点信息列表
     * @param node_counter 节点计数器
     */
    void extractRegionNodesEnhanced(std::vector<NodeInfo>& nodes, int& node_counter);
    
    /**
     * @brief 增强版本的odom节点提取
     * @param nodes 输出的节点信息列表
     * @param node_counter 节点计数器
     */
    void extractOdomNodeEnhanced(std::vector<NodeInfo>& nodes, int& node_counter);
    
    /**
     * @brief 增强版本的viewpoint节点提取
     * @param nodes 输出的节点信息列表
     * @param node_counter 节点计数器
     */
    void extractViewpointNodesEnhanced(std::vector<NodeInfo>& nodes, int& node_counter);
};

} // namespace fast_planner

#endif // TOPO_GRAPH_INTEGRATED_EXTRACTOR_H
