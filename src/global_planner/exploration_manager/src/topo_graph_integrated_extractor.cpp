/**
 * @file topo_graph_integrated_extractor.cpp
 * @brief 集成的拓扑图提取器实现
 */

#include <epic_planner/topo_graph_integrated_extractor.h>
#include <fstream>
#include <iomanip>

namespace fast_planner {

void TopoGraphExtractorInline::exportCompleteGraph(TopoGraph::Ptr topo_graph, 
                                                   std::vector<NodeInfo>& nodes, 
                                                   std::vector<EdgeInfo>& edges) {
    if (!topo_graph) {
        ROS_WARN("TopoGraph is null");
        return;
    }

    nodes.clear();
    edges.clear();
    int node_counter = 0;
    int edge_counter = 0;

    // 提取各种类型的节点（不包含reachability和TSP信息）
    extractRegionNodes(topo_graph, nodes, node_counter);
    extractOdomNode(topo_graph, nodes, node_counter);
    extractViewpointNodes(topo_graph, nodes, node_counter);

    // 生成边连接信息
    generateEdges(nodes, edges, edge_counter);

    ROS_INFO("Extracted %zu nodes and %zu edges from topo graph", 
            nodes.size(), edges.size());
}

void TopoGraphExtractorInline::extractRegionNodes(TopoGraph::Ptr topo_graph, 
                                                  std::vector<NodeInfo>& nodes, 
                                                  int& node_counter) {
    // 静态调试标志，可以通过ROS参数控制
    static bool debug_enabled = []() {
        bool debug = false;
        ros::param::param("/exploration_node/topo_extraction/debug_output", debug, false);
        return debug;
    }();
    
    if (debug_enabled) {
        ROS_INFO("=== Extracting Region Nodes ===");
        ROS_INFO("Total regions_arr_ size: %d", (int)topo_graph->regions_arr_.size());
        ROS_INFO("Total toponodes_update_region_arr_ size: %d", (int)topo_graph->toponodes_update_region_arr_.size());
        ROS_INFO("Total reg_map_idx2ptr_ size: %d", (int)topo_graph->reg_map_idx2ptr_.size());
        ROS_INFO("Total history_odom_nodes_ size: %d", (int)topo_graph->history_odom_nodes_.size());
    }
    
    // 提取所有区域节点中的拓扑节点
    int regions_with_nodes = 0;
    int total_topo_nodes = 0;
    int total_viewpoints = 0;
    
    // 检查 regions_arr_
    for (size_t i = 0; i < topo_graph->regions_arr_.size(); ++i) {
        auto region = topo_graph->regions_arr_[i];
        if (!region) continue;

        if (debug_enabled && !region->topo_nodes_.empty()) {
            ROS_INFO("Region %d has %d topo nodes", (int)i, (int)region->topo_nodes_.size());
        }
        if (!region->topo_nodes_.empty()) regions_with_nodes++;
        
        // 从RegionNode中提取TopoNode
        for (auto topo_node : region->topo_nodes_) {
            if (!topo_node) continue;
            
            total_topo_nodes++;
            
            NodeInfo node_info;
            node_info.node_id = node_counter++;
            node_info.position = topo_node->center_.cast<double>();
            node_info.yaw = static_cast<double>(topo_node->yaw_);
            node_info.is_viewpoint = topo_node->is_viewpoint_;
            node_info.is_current_odom = false;
            node_info.is_history_odom = topo_node->is_history_odom_node_;
            node_info.region_id = static_cast<int>(i);
            
            // 默认值初始化（静态方法版本不包含这些信息）
            node_info.is_reachable = false;
            node_info.tsp_order_index = -1;
            node_info.reachable_distance = -1.0;
            
            nodes.push_back(node_info);
            
            if (topo_node->is_viewpoint_) {
                total_viewpoints++;
                if (debug_enabled) {
                    ROS_INFO("⭐ FOUND VIEWPOINT in regions_arr_[%zu]: node %d", i, node_info.node_id);
                }
            }
            
            if (debug_enabled) {
                ROS_DEBUG("Added region node %d: pos(%.2f,%.2f,%.2f), viewpoint=%s, history_odom=%s", 
                         node_info.node_id, 
                         node_info.position.x(), node_info.position.y(), node_info.position.z(),
                         node_info.is_viewpoint ? "true" : "false",
                         node_info.is_history_odom ? "true" : "false");
            }
        }
    }
    
    // 检查 toponodes_update_region_arr_
    for (size_t i = 0; i < topo_graph->toponodes_update_region_arr_.size(); ++i) {
        auto region = topo_graph->toponodes_update_region_arr_[i];
        if (!region) continue;

        if (debug_enabled && !region->topo_nodes_.empty()) {
            ROS_INFO("Toponodes region %d has %d topo_nodes", (int)i, (int)region->topo_nodes_.size());
        }
        
        for (auto topo_node : region->topo_nodes_) {
            if (!topo_node) continue;
            
            NodeInfo node_info;
            node_info.node_id = node_counter++;
            node_info.position = topo_node->center_.cast<double>();
            node_info.yaw = static_cast<double>(topo_node->yaw_);
            node_info.is_viewpoint = topo_node->is_viewpoint_;
            node_info.is_current_odom = false;
            node_info.is_history_odom = topo_node->is_history_odom_node_;
            node_info.region_id = static_cast<int>(i + 1000); // 区别于regions_arr_
            
            // 默认值初始化（静态方法版本不包含这些信息）
            node_info.is_reachable = false;
            node_info.tsp_order_index = -1;
            node_info.reachable_distance = -1.0;
            
            nodes.push_back(node_info);
            total_topo_nodes++;
            
            if (topo_node->is_viewpoint_) {
                total_viewpoints++;
                if (debug_enabled) {
                    ROS_INFO("⭐ FOUND VIEWPOINT in toponodes_region[%zu]: node %d", i, node_info.node_id);
                }
            }
            
            if (debug_enabled) {
                ROS_DEBUG("Added toponode region node %d: pos(%.2f,%.2f,%.2f), viewpoint=%s", 
                         node_info.node_id, 
                         node_info.position.x(), node_info.position.y(), node_info.position.z(),
                         node_info.is_viewpoint ? "true" : "false");
            }
        }
    }
    
    // 检查 reg_map_idx2ptr_
    int map_regions_with_nodes = 0;
    for (auto& pair : topo_graph->reg_map_idx2ptr_) {
        auto region = pair.second;
        if (!region || region->topo_nodes_.empty()) continue;
        
        if (debug_enabled) {
            ROS_INFO("Map region [%d,%d,%d] has %d topo_nodes", 
                    pair.first.x(), pair.first.y(), pair.first.z(),
                    (int)region->topo_nodes_.size());
        }
        
        for (auto topo_node : region->topo_nodes_) {
            if (!topo_node) continue;
            
            NodeInfo node_info;
            node_info.node_id = node_counter++;
            node_info.position = topo_node->center_.cast<double>();
            node_info.yaw = static_cast<double>(topo_node->yaw_);
            node_info.is_viewpoint = topo_node->is_viewpoint_;
            node_info.is_current_odom = false;
            node_info.is_history_odom = topo_node->is_history_odom_node_;
            node_info.region_id = static_cast<int>(pair.first.x() * 10000 + pair.first.y() * 100 + pair.first.z()); 
            
            // 默认值初始化（静态方法版本不包含这些信息）
            node_info.is_reachable = false;
            node_info.tsp_order_index = -1;
            node_info.reachable_distance = -1.0;
            
            nodes.push_back(node_info);
            total_topo_nodes++;
            
            if (topo_node->is_viewpoint_) {
                total_viewpoints++;
                if (debug_enabled) {
                    ROS_INFO("⭐ FOUND VIEWPOINT in map region [%d,%d,%d]: node %d", 
                             pair.first.x(), pair.first.y(), pair.first.z(), node_info.node_id);
                }
            }
            
            if (debug_enabled) {
                ROS_DEBUG("Added map region node %d: pos(%.2f,%.2f,%.2f), viewpoint=%s", 
                         node_info.node_id, 
                         node_info.position.x(), node_info.position.y(), node_info.position.z(),
                         node_info.is_viewpoint ? "true" : "false");
            }
        }
        map_regions_with_nodes++;
    }
    
    // 检查 history_odom_nodes_
    for (size_t i = 0; i < topo_graph->history_odom_nodes_.size(); ++i) {
        auto topo_node = topo_graph->history_odom_nodes_[i];
        if (!topo_node) continue;
        
        NodeInfo node_info;
        node_info.node_id = node_counter++;
        node_info.position = topo_node->center_.cast<double>();
        node_info.yaw = static_cast<double>(topo_node->yaw_);
        node_info.is_viewpoint = topo_node->is_viewpoint_;
        node_info.is_current_odom = false;
        node_info.is_history_odom = true;
        node_info.region_id = -10 - static_cast<int>(i); // 特殊标记为历史odom
        
        // 默认值初始化（静态方法版本不包含这些信息）
        node_info.is_reachable = false;
        node_info.tsp_order_index = -1;
        node_info.reachable_distance = -1.0;
        
        nodes.push_back(node_info);
        total_topo_nodes++;
        
        if (topo_node->is_viewpoint_) {
            total_viewpoints++;
            if (debug_enabled) {
                ROS_INFO("⭐ FOUND VIEWPOINT in history_odom[%zu]: node %d", i, node_info.node_id);
            }
        }
        
        if (debug_enabled) {
            ROS_DEBUG("Added history odom node %d: pos(%.2f,%.2f,%.2f)", 
                     node_info.node_id, 
                     node_info.position.x(), node_info.position.y(), node_info.position.z());
        }
    }
    
    // 总结信息（总是输出）
    ROS_INFO("Extracted %d regions (%d with nodes), %d topo nodes, %d viewpoints", 
             (int)topo_graph->regions_arr_.size(), regions_with_nodes, total_topo_nodes, total_viewpoints);
    
    if (debug_enabled) {
        ROS_INFO("🎯 VIEWPOINT SUMMARY: Found %d viewpoints out of %d total nodes", total_viewpoints, total_topo_nodes);
    }
}

void TopoGraphExtractorInline::extractOdomNode(TopoGraph::Ptr topo_graph, 
                                               std::vector<NodeInfo>& nodes, 
                                               int& node_counter) {
    // 静态调试标志
    static bool debug_enabled = []() {
        bool debug = false;
        ros::param::param("/exploration_node/topo_extraction/debug_output", debug, false);
        return debug;
    }();
    
    if (debug_enabled) {
        ROS_INFO("=== Extracting Odom Node ===");
    }
    
    // 提取当前位置节点
    if (topo_graph->odom_node_) {
        NodeInfo odom_node;
        odom_node.node_id = node_counter++;
        odom_node.position = topo_graph->odom_node_->center_.cast<double>();
        odom_node.yaw = static_cast<double>(topo_graph->odom_node_->yaw_);
        odom_node.is_viewpoint = topo_graph->odom_node_->is_viewpoint_;  // 使用实际的viewpoint状态
        odom_node.is_current_odom = true;
        odom_node.is_history_odom = false;
        odom_node.region_id = -1; // 特殊标记为当前位置
        
        // 默认值初始化（静态方法版本不包含这些信息）
        odom_node.is_reachable = true; // 当前位置总是可达的
        odom_node.tsp_order_index = 0; // 在TSP中，索引0通常代表起始位置
        odom_node.reachable_distance = 0.0; // 到自身的距离为0
        
        nodes.push_back(odom_node);
        
        if (topo_graph->odom_node_->is_viewpoint_) {
            ROS_INFO("⭐ Current odom node %d is a VIEWPOINT at pos(%.2f,%.2f,%.2f)", 
                     odom_node.node_id,
                     odom_node.position.x(), odom_node.position.y(), odom_node.position.z());
        } else if (debug_enabled) {
            ROS_INFO("Added current odom node %d: pos(%.2f,%.2f,%.2f), yaw=%.2f (not viewpoint)", 
                     odom_node.node_id,
                     odom_node.position.x(), odom_node.position.y(), odom_node.position.z(),
                     odom_node.yaw);
        }
    } else {
        ROS_WARN("No current odom node available");
    }
}

void TopoGraphExtractorInline::extractViewpointNodes(TopoGraph::Ptr topo_graph, 
                                                     std::vector<NodeInfo>& nodes, 
                                                     int& node_counter) {
    // 静态调试标志
    static bool debug_enabled = []() {
        bool debug = false;
        ros::param::param("/exploration_node/topo_extraction/debug_output", debug, false);
        return debug;
    }();
    
    if (debug_enabled) {
        ROS_INFO("=== Extracting Viewpoint Nodes ===");
        ROS_INFO("Total viewpoints_update_region_arr_ size: %d", (int)topo_graph->viewpoints_update_region_arr_.size());
    }
    
    // 提取视点节点 - 从viewpoints_update_region_arr_中的TopoNode提取
    int regions_with_viewpoints = 0;
    int total_viewpoint_nodes = 0;
    
    for (auto viewpoint_region : topo_graph->viewpoints_update_region_arr_) {
        if (!viewpoint_region) continue;
        
        if (debug_enabled && !viewpoint_region->topo_nodes_.empty()) {
            ROS_INFO("Viewpoint region has %d topo nodes", (int)viewpoint_region->topo_nodes_.size());
        }
        regions_with_viewpoints++;
        
        // 从RegionNode中提取TopoNode
        for (auto topo_node : viewpoint_region->topo_nodes_) {
            if (!topo_node || !topo_node->is_viewpoint_) continue;
            
            total_viewpoint_nodes++;
            
            NodeInfo vp_node;
            vp_node.node_id = node_counter++;
            vp_node.position = topo_node->center_.cast<double>();
            vp_node.yaw = static_cast<double>(topo_node->yaw_);
            vp_node.is_viewpoint = true;
            vp_node.is_current_odom = false;
            vp_node.is_history_odom = false;
            vp_node.region_id = -2; // 特殊标记为视点
            
            // 默认值初始化（静态方法版本不包含这些信息）
            vp_node.is_reachable = false;
            vp_node.tsp_order_index = -1;
            vp_node.reachable_distance = -1.0;
            
            nodes.push_back(vp_node);
            
            if (debug_enabled) {
                ROS_INFO("Added viewpoint node %d: pos(%.2f,%.2f,%.2f), yaw=%.2f", 
                         vp_node.node_id,
                         vp_node.position.x(), vp_node.position.y(), vp_node.position.z(),
                         vp_node.yaw);
            }
        }
    }
    
    if (total_viewpoint_nodes > 0) {
        ROS_INFO("Extracted %d dedicated viewpoint nodes from %d regions", 
                 total_viewpoint_nodes, regions_with_viewpoints);
    }
}

void TopoGraphExtractorInline::generateEdges(const std::vector<NodeInfo>& nodes, 
                                             std::vector<EdgeInfo>& edges, 
                                             int& edge_counter,
                                             double distance_threshold) {
    // 基于距离创建连接关系
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (size_t j = i + 1; j < nodes.size(); ++j) {
            double dist = (nodes[i].position - nodes[j].position).norm();
            if (dist < distance_threshold) {
                EdgeInfo edge;
                edge.edge_id = edge_counter++;
                edge.from_node_id = nodes[i].node_id;
                edge.to_node_id = nodes[j].node_id;
                edge.weight = dist;
                edge.is_reachable = true; // 简化假设所有在阈值内的都可达
                edges.push_back(edge);
            }
        }
    }
    ROS_DEBUG("Generated %d edges with distance threshold %.2f", 
              edge_counter, distance_threshold);
}

// TopoExtractorIntegrated 类实现

TopoExtractorIntegrated::TopoExtractorIntegrated(TopoGraph::Ptr graph, 
                                               const std::string& export_dir,
                                               bool enable_extraction,
                                               double distance_threshold) 
    : topo_graph_(graph), export_dir_(export_dir), enable_extraction_(enable_extraction), 
      distance_threshold_(distance_threshold), debug_output_(false), nh_ptr_(nullptr) {
    
    ROS_INFO("=== TopoExtractorIntegrated Constructor (Direct Mode) ===");
    ROS_INFO("Export directory: '%s'", export_dir_.c_str());
    ROS_INFO("Extraction enabled: %s", enable_extraction_ ? "true" : "false");
    ROS_INFO("Distance threshold: %.2f", distance_threshold_);
    ROS_INFO("Debug output: %s", debug_output_ ? "enabled" : "disabled");
    
    // 确保输出目录存在
    if (!export_dir_.empty()) {
        std::string mkdir_cmd = "mkdir -p " + export_dir_;
        system(mkdir_cmd.c_str());
        ROS_INFO("Ensured output directory exists: %s", export_dir_.c_str());
    }
}

TopoExtractorIntegrated::TopoExtractorIntegrated(ros::NodeHandle& nh, TopoGraph::Ptr graph,
                                               bool enable_timer, double extraction_rate) 
    : topo_graph_(graph), nh_ptr_(&nh) {
    
    ROS_INFO("=== TopoExtractorIntegrated Constructor (ROS Mode) ===");
    ROS_INFO("NodeHandle namespace: '%s'", nh.getNamespace().c_str());
    
    // 初始化参数
    initializeParams(&nh);
    
    if (enable_timer && enable_extraction_) {
        // 设置定时器
        extract_timer_ = nh.createTimer(ros::Duration(1.0/extraction_rate),
                                       &TopoExtractorIntegrated::extractCallback, this);
        ROS_INFO("✓ Topo extraction timer ENABLED at %.2f Hz", extraction_rate);
    }
    
    // 设置服务
    extract_service_ = nh.advertiseService("/extract_topo_graph",
                                          &TopoExtractorIntegrated::extractService, this);
    ROS_INFO("✓ Topo graph extraction service '/extract_topo_graph' available");
}

void TopoExtractorIntegrated::initializeParams(ros::NodeHandle* nh) {
    if (nh) {
        // 从ROS参数服务器读取参数
        nh->param("topo_extraction/enable", enable_extraction_, false);
        nh->param("topo_extraction/export_dir", export_dir_, std::string("/tmp"));
        nh->param("topo_extraction/distance_threshold", distance_threshold_, 10.0);
        nh->param("topo_extraction/debug_output", debug_output_, false);
        
        ROS_INFO("Parameters read from ROS:");
        ROS_INFO("  enable_extraction_: %s", enable_extraction_ ? "true" : "false");
        ROS_INFO("  export_dir_: '%s'", export_dir_.c_str());
        ROS_INFO("  distance_threshold_: %.2f", distance_threshold_);
        ROS_INFO("  debug_output_: %s", debug_output_ ? "true" : "false");
    } else {
        // 使用默认值
        enable_extraction_ = true;
        export_dir_ = "/tmp";
        distance_threshold_ = 10.0;
        debug_output_ = false;
        
        ROS_INFO("Using default parameters:");
        ROS_INFO("  enable_extraction_: %s", enable_extraction_ ? "true" : "false");
        ROS_INFO("  export_dir_: '%s'", export_dir_.c_str());
        ROS_INFO("  distance_threshold_: %.2f", distance_threshold_);
        ROS_INFO("  debug_output_: %s", debug_output_ ? "true" : "false");
    }
    
    // 确保输出目录存在
    if (!export_dir_.empty()) {
        std::string mkdir_cmd = "mkdir -p " + export_dir_;
        system(mkdir_cmd.c_str());
        ROS_INFO("Ensured output directory exists: %s", export_dir_.c_str());
    }
}

void TopoExtractorIntegrated::extractCallback(const ros::TimerEvent& event) {
    ROS_INFO("=== TopoExtractor Timer Callback Triggered ===");
    extractTopoGraph("timer_callback");
}

bool TopoExtractorIntegrated::extractService(std_srvs::Trigger::Request& req, 
                                            std_srvs::Trigger::Response& res) {
    try {
        bool success = extractTopoGraph("service_call");
        res.success = success;
        res.message = success ? "Topo graph extracted successfully" : "Failed to extract topo graph";
        return true;
    } catch (const std::exception& e) {
        res.success = false;
        res.message = "Failed to extract topo graph: " + std::string(e.what());
        return true;
    }
}

bool TopoExtractorIntegrated::extractTopoGraph(const std::string& context) {
    if (!enable_extraction_) {
        if (debug_output_) ROS_DEBUG("Topo extraction disabled, skipping...");
        return false;
    }
    
    if (debug_output_) {
        ROS_INFO("=== TopoExtractorIntegrated::extractTopoGraph() called [%s] ===", context.c_str());
    }
    
    if (!topo_graph_) {
        ROS_WARN("Topo graph not available (null pointer)");
        return false;
    }
    
    if (debug_output_) {
        ROS_INFO("Topo graph available, starting enhanced extraction...");
    }
    
    try {
        std::vector<NodeInfo> nodes;
        std::vector<EdgeInfo> edges;
        
        // 使用增强版本的提取方法
        extractCompleteGraphEnhanced(nodes, edges);
        
        if (nodes.empty()) {
            ROS_WARN("No nodes extracted from topo graph");
            return false;
        }
        
        // 生成文件名
        std::string timestamp = std::to_string(ros::Time::now().toSec());
        std::string filename = export_dir_ + "/topo_graph_" + timestamp + ".txt";
        
        // 导出
        exportSimpleFormat(nodes, edges, filename);
        ROS_INFO("✓ Exported topo graph [%s]: %zu nodes, %zu edges to %s", 
                 context.c_str(), nodes.size(), edges.size(), filename.c_str());
        return true;
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to extract topo graph [%s]: %s", context.c_str(), e.what());
        return false;
    }
}

void TopoExtractorIntegrated::extractCompleteGraphEnhanced(std::vector<NodeInfo>& nodes, 
                                                          std::vector<EdgeInfo>& edges) {
    if (!topo_graph_) {
        ROS_WARN("TopoGraph is null");
        return;
    }

    nodes.clear();
    edges.clear();
    int node_counter = 0;
    int edge_counter = 0;

    // 提取各种类型的节点（使用增强版本，包含reachability和TSP信息）
    extractRegionNodesEnhanced(nodes, node_counter);
    extractOdomNodeEnhanced(nodes, node_counter);
    extractViewpointNodesEnhanced(nodes, node_counter);

    // 生成边连接信息
    TopoGraphExtractorInline::generateEdges(nodes, edges, edge_counter, distance_threshold_);

    ROS_INFO("Enhanced extraction completed: %zu nodes, %zu edges", nodes.size(), edges.size());
}

void TopoExtractorIntegrated::exportSimpleFormat(const std::vector<NodeInfo>& nodes,
                                                 const std::vector<EdgeInfo>& edges,
                                                 const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        ROS_ERROR("Failed to open file for writing: %s", filename.c_str());
        return;
    }
    
    // 写入文件头信息
    file << "# Topo Graph Export with Exploration Statistics - " << ros::Time::now() << std::endl;
    file << "# Generated by EPIC TopoGraphExtractor" << std::endl;
    file << "# Total nodes: " << nodes.size() << ", Total edges: " << edges.size() << std::endl;
    file << std::endl;
    
    // 写入探索统计信息
    if (exploration_stats_) {
        file << "# ===== EXPLORATION STATISTICS =====" << std::endl;
        file << exploration_stats_->exportMetricsString() << std::endl;
    } else {
        file << "# No exploration statistics available" << std::endl << std::endl;
    }
    
    // 写入节点信息
    file << "# 节点格式: node_id x y z yaw is_viewpoint is_current is_history region_id is_reachable tsp_order_index distance observation_score cluster_distance" << std::endl;
    file << "NODES" << std::endl;
    for (const auto& node : nodes) {
        file << node.node_id << " "
             << std::fixed << std::setprecision(6) 
             << node.position.x() << " " << node.position.y() << " " << node.position.z() << " "
             << node.yaw << " "
             << (node.is_viewpoint ? 1 : 0) << " "
             << (node.is_current_odom ? 1 : 0) << " "
             << (node.is_history_odom ? 1 : 0) << " "
             << node.region_id << " "
             << (node.is_reachable ? 1 : 0) << " "
             << node.tsp_order_index << " "
             << std::fixed << std::setprecision(6) << node.reachable_distance << " "
             << node.observation_score << " "
             << node.cluster_distance << std::endl;
    }
    
    // 写入边信息
    file << std::endl << "# 边格式: edge_id from_node_id to_node_id weight is_reachable" << std::endl;
    file << "EDGES" << std::endl;
    for (const auto& edge : edges) {
        file << edge.edge_id << " "
             << edge.from_node_id << " " << edge.to_node_id << " "
             << std::fixed << std::setprecision(6) << edge.weight << " "
             << (edge.is_reachable ? 1 : 0) << std::endl;
    }
    
    file.close();
    ROS_DEBUG("Topo graph data with exploration statistics written to: %s", filename.c_str());
}

void TopoExtractorIntegrated::updateViewpointInfo(const std::vector<TopoNode::Ptr>& reachable_viewpoints,
                                                  const std::vector<double>& reachable_distances,
                                                  const std::vector<int>& tsp_indices,
                                                  const std::vector<ViewpointBenefit>& viewpoint_benefits) {
    // 清空之前的信息
    viewpoint_reachability_map_.clear();
    viewpoint_distance_map_.clear();
    viewpoint_tsp_index_map_.clear();
    viewpoint_observation_score_map_.clear();
    viewpoint_cluster_distance_map_.clear();
    
    if (debug_output_) {
        ROS_INFO("=== Updating Simplified Viewpoint Info ===");
        ROS_INFO("Reachable viewpoints: %zu", reachable_viewpoints.size());
        ROS_INFO("TSP indices size: %zu", tsp_indices.size());
        ROS_INFO("Viewpoint benefits size: %zu", viewpoint_benefits.size());
    }
    
    // 记录可达的viewpoints基本信息
    for (size_t i = 0; i < reachable_viewpoints.size(); ++i) {
        auto vp = reachable_viewpoints[i];
        if (!vp) continue;
        
        viewpoint_reachability_map_[vp] = true;
        if (i < reachable_distances.size()) {
            viewpoint_distance_map_[vp] = reachable_distances[i];
        }
    }
    
    // 记录收益信息
    for (const auto& benefit : viewpoint_benefits) {
        if (!benefit.viewpoint) continue;
        
        viewpoint_observation_score_map_[benefit.viewpoint] = benefit.observation_score;
        viewpoint_cluster_distance_map_[benefit.viewpoint] = benefit.cluster_distance;
        
        if (debug_output_) {
            ROS_INFO("VP benefit: obs_score=%.1f, cluster_dist=%.2f, reachable=%s",
                     benefit.observation_score, benefit.cluster_distance,
                     benefit.is_reachable ? "YES" : "NO");
        }
    }
    
    ROS_INFO("TopoExtractor: Updated %zu observation scores, %zu cluster distances", 
             viewpoint_observation_score_map_.size(), viewpoint_cluster_distance_map_.size());
    
    // 记录TSP顺序信息
    for (size_t i = 0; i < tsp_indices.size(); ++i) {
        int tsp_idx = tsp_indices[i];
        if (tsp_idx > 0 && tsp_idx <= (int)reachable_viewpoints.size()) {
            TopoNode::Ptr vp = reachable_viewpoints[tsp_idx - 1];
            viewpoint_tsp_index_map_[vp] = static_cast<int>(i);
            
            if (debug_output_) {
                ROS_DEBUG("Viewpoint %p TSP order: %d (original TSP index: %d)", 
                         vp.get(), static_cast<int>(i), tsp_idx);
            }
        }
    }
    
    if (debug_output_) {
        ROS_INFO("✓ Updated info for %zu reachable viewpoints with benefits", 
                 viewpoint_reachability_map_.size());
    }
}

void TopoExtractorIntegrated::extractRegionNodesEnhanced(std::vector<NodeInfo>& nodes, int& node_counter) {
    // 静态调试标志，可以通过ROS参数控制
    static bool debug_enabled = []() {
        bool debug = false;
        ros::param::param("/exploration_node/topo_extraction/debug_output", debug, false);
        return debug;
    }();
    
    if (debug_enabled) {
        ROS_INFO("=== Enhanced Extracting Region Nodes ===");
        ROS_INFO("Total regions_arr_ size: %d", (int)topo_graph_->regions_arr_.size());
        ROS_INFO("Total toponodes_update_region_arr_ size: %d", (int)topo_graph_->toponodes_update_region_arr_.size());
        ROS_INFO("Total reg_map_idx2ptr_ size: %d", (int)topo_graph_->reg_map_idx2ptr_.size());
        ROS_INFO("Total history_odom_nodes_ size: %d", (int)topo_graph_->history_odom_nodes_.size());
    }
    
    // 提取所有区域节点中的拓扑节点
    int regions_with_nodes = 0;
    int total_topo_nodes = 0;
    int total_viewpoints = 0;
    
    // 检查 regions_arr_
    for (size_t i = 0; i < topo_graph_->regions_arr_.size(); ++i) {
        auto region = topo_graph_->regions_arr_[i];
        if (!region) continue;

        if (debug_enabled && !region->topo_nodes_.empty()) {
            ROS_INFO("Region %d has %d topo nodes", (int)i, (int)region->topo_nodes_.size());
        }
        if (!region->topo_nodes_.empty()) regions_with_nodes++;
        
        // 从RegionNode中提取TopoNode
        for (auto topo_node : region->topo_nodes_) {
            if (!topo_node) continue;
            
            total_topo_nodes++;
            
            NodeInfo node_info;
            node_info.node_id = node_counter++;
            node_info.position = topo_node->center_.cast<double>();
            node_info.yaw = static_cast<double>(topo_node->yaw_);
            node_info.is_viewpoint = topo_node->is_viewpoint_;
            node_info.is_current_odom = false;
            node_info.is_history_odom = topo_node->is_history_odom_node_;
            node_info.region_id = static_cast<int>(i);
            
            // 增强字段初始化
            node_info.is_reachable = viewpoint_reachability_map_.count(topo_node) > 0;
            node_info.tsp_order_index = viewpoint_tsp_index_map_.count(topo_node) > 0 ? 
                                       viewpoint_tsp_index_map_[topo_node] : -1;
            node_info.reachable_distance = viewpoint_distance_map_.count(topo_node) > 0 ?
                                          viewpoint_distance_map_[topo_node] : -1.0;
            
            // 收益信息字段 - 为所有节点设置收益信息（视点和非视点都设置）
            node_info.observation_score = viewpoint_observation_score_map_.count(topo_node) > 0 ?
                                         viewpoint_observation_score_map_[topo_node] : 0.0;
            node_info.cluster_distance = viewpoint_cluster_distance_map_.count(topo_node) > 0 ?
                                        viewpoint_cluster_distance_map_[topo_node] : -1.0;
            
            nodes.push_back(node_info);
            
            if (topo_node->is_viewpoint_) {
                total_viewpoints++;
                if (debug_enabled) {
                    ROS_INFO("⭐ FOUND VIEWPOINT in regions_arr_[%zu]: node %d (reachable: %s, TSP idx: %d, obs_score: %.1f, cluster_dist: %.2f)", 
                             i, node_info.node_id, node_info.is_reachable ? "YES" : "NO", node_info.tsp_order_index,
                             node_info.observation_score, node_info.cluster_distance);
                }
            }
            
            if (debug_enabled) {
                ROS_DEBUG("Added region node %d: pos(%.2f,%.2f,%.2f), viewpoint=%s, history_odom=%s", 
                         node_info.node_id, 
                         node_info.position.x(), node_info.position.y(), node_info.position.z(),
                         node_info.is_viewpoint ? "true" : "false",
                         node_info.is_history_odom ? "true" : "false");
            }
        }
    }
    
    // 检查 toponodes_update_region_arr_
    for (size_t i = 0; i < topo_graph_->toponodes_update_region_arr_.size(); ++i) {
        auto region = topo_graph_->toponodes_update_region_arr_[i];
        if (!region) continue;

        if (debug_enabled && !region->topo_nodes_.empty()) {
            ROS_INFO("Toponodes region %d has %d topo_nodes", (int)i, (int)region->topo_nodes_.size());
        }
        
        for (auto topo_node : region->topo_nodes_) {
            if (!topo_node) continue;
            
            NodeInfo node_info;
            node_info.node_id = node_counter++;
            node_info.position = topo_node->center_.cast<double>();
            node_info.yaw = static_cast<double>(topo_node->yaw_);
            node_info.is_viewpoint = topo_node->is_viewpoint_;
            node_info.is_current_odom = false;
            node_info.is_history_odom = topo_node->is_history_odom_node_;
            node_info.region_id = static_cast<int>(i + 1000); // 区别于regions_arr_
            
            // 增强字段初始化
            node_info.is_reachable = viewpoint_reachability_map_.count(topo_node) > 0;
            node_info.tsp_order_index = viewpoint_tsp_index_map_.count(topo_node) > 0 ? 
                                       viewpoint_tsp_index_map_[topo_node] : -1;
            node_info.reachable_distance = viewpoint_distance_map_.count(topo_node) > 0 ?
                                          viewpoint_distance_map_[topo_node] : -1.0;
            
            // 收益信息字段 - 为所有节点设置收益信息（视点和非视点都设置）
            node_info.observation_score = viewpoint_observation_score_map_.count(topo_node) > 0 ?
                                         viewpoint_observation_score_map_[topo_node] : 0.0;
            node_info.cluster_distance = viewpoint_cluster_distance_map_.count(topo_node) > 0 ?
                                        viewpoint_cluster_distance_map_[topo_node] : -1.0;
            
            nodes.push_back(node_info);
            total_topo_nodes++;
            
            if (topo_node->is_viewpoint_) {
                total_viewpoints++;
                if (debug_enabled) {
                    ROS_INFO("⭐ FOUND VIEWPOINT in toponodes_region[%zu]: node %d (reachable: %s, TSP idx: %d, obs_score: %.1f, cluster_dist: %.2f)", 
                             i, node_info.node_id, node_info.is_reachable ? "YES" : "NO", node_info.tsp_order_index,
                             node_info.observation_score, node_info.cluster_distance);
                }
            }
            
            if (debug_enabled) {
                ROS_DEBUG("Added toponode region node %d: pos(%.2f,%.2f,%.2f), viewpoint=%s", 
                         node_info.node_id, 
                         node_info.position.x(), node_info.position.y(), node_info.position.z(),
                         node_info.is_viewpoint ? "true" : "false");
            }
        }
    }
    
    // 检查 reg_map_idx2ptr_
    int map_regions_with_nodes = 0;
    for (auto& pair : topo_graph_->reg_map_idx2ptr_) {
        auto region = pair.second;
        if (!region || region->topo_nodes_.empty()) continue;
        
        if (debug_enabled) {
            ROS_INFO("Map region [%d,%d,%d] has %d topo_nodes", 
                    pair.first.x(), pair.first.y(), pair.first.z(),
                    (int)region->topo_nodes_.size());
        }
        
        for (auto topo_node : region->topo_nodes_) {
            if (!topo_node) continue;
            
            NodeInfo node_info;
            node_info.node_id = node_counter++;
            node_info.position = topo_node->center_.cast<double>();
            node_info.yaw = static_cast<double>(topo_node->yaw_);
            node_info.is_viewpoint = topo_node->is_viewpoint_;
            node_info.is_current_odom = false;
            node_info.is_history_odom = topo_node->is_history_odom_node_;
            node_info.region_id = static_cast<int>(pair.first.x() * 10000 + pair.first.y() * 100 + pair.first.z()); 
            
            // 增强字段初始化
            node_info.is_reachable = viewpoint_reachability_map_.count(topo_node) > 0;
            node_info.tsp_order_index = viewpoint_tsp_index_map_.count(topo_node) > 0 ? 
                                       viewpoint_tsp_index_map_[topo_node] : -1;
            node_info.reachable_distance = viewpoint_distance_map_.count(topo_node) > 0 ?
                                          viewpoint_distance_map_[topo_node] : -1.0;
            
            // 收益信息字段 - 为所有节点设置收益信息（视点和非视点都设置）
            node_info.observation_score = viewpoint_observation_score_map_.count(topo_node) > 0 ?
                                         viewpoint_observation_score_map_[topo_node] : 0.0;
            node_info.cluster_distance = viewpoint_cluster_distance_map_.count(topo_node) > 0 ?
                                        viewpoint_cluster_distance_map_[topo_node] : -1.0;
            
            nodes.push_back(node_info);
            total_topo_nodes++;
            
            if (topo_node->is_viewpoint_) {
                total_viewpoints++;
                if (debug_enabled) {
                    ROS_INFO("⭐ FOUND VIEWPOINT in map region [%d,%d,%d]: node %d (reachable: %s, TSP idx: %d, obs_score: %.1f, cluster_dist: %.2f)", 
                             pair.first.x(), pair.first.y(), pair.first.z(), node_info.node_id,
                             node_info.is_reachable ? "YES" : "NO", node_info.tsp_order_index,
                             node_info.observation_score, node_info.cluster_distance);
                }
            }
            
            if (debug_enabled) {
                ROS_DEBUG("Added map region node %d: pos(%.2f,%.2f,%.2f), viewpoint=%s", 
                         node_info.node_id, 
                         node_info.position.x(), node_info.position.y(), node_info.position.z(),
                         node_info.is_viewpoint ? "true" : "false");
            }
        }
        map_regions_with_nodes++;
    }
    
    // 检查 history_odom_nodes_
    for (size_t i = 0; i < topo_graph_->history_odom_nodes_.size(); ++i) {
        auto topo_node = topo_graph_->history_odom_nodes_[i];
        if (!topo_node) continue;
        
        NodeInfo node_info;
        node_info.node_id = node_counter++;
        node_info.position = topo_node->center_.cast<double>();
        node_info.yaw = static_cast<double>(topo_node->yaw_);
        node_info.is_viewpoint = topo_node->is_viewpoint_;
        node_info.is_current_odom = false;
        node_info.is_history_odom = true;
        node_info.region_id = -10 - static_cast<int>(i); // 特殊标记为历史odom
        
        // 增强字段初始化
        node_info.is_reachable = viewpoint_reachability_map_.count(topo_node) > 0;
        node_info.tsp_order_index = viewpoint_tsp_index_map_.count(topo_node) > 0 ? 
                                   viewpoint_tsp_index_map_[topo_node] : -1;
        node_info.reachable_distance = viewpoint_distance_map_.count(topo_node) > 0 ?
                                      viewpoint_distance_map_[topo_node] : -1.0;
        
        // 收益信息字段 - 为所有节点设置收益信息（视点和非视点都设置）
        node_info.observation_score = viewpoint_observation_score_map_.count(topo_node) > 0 ?
                                     viewpoint_observation_score_map_[topo_node] : 0.0;
        node_info.cluster_distance = viewpoint_cluster_distance_map_.count(topo_node) > 0 ?
                                    viewpoint_cluster_distance_map_[topo_node] : -1.0;
        
        nodes.push_back(node_info);
        total_topo_nodes++;
        
        if (topo_node->is_viewpoint_) {
            total_viewpoints++;
            if (debug_enabled) {
                ROS_INFO("⭐ FOUND VIEWPOINT in history_odom[%zu]: node %d (reachable: %s, TSP idx: %d, obs_score: %.1f, cluster_dist: %.2f)", 
                         i, node_info.node_id, node_info.is_reachable ? "YES" : "NO", node_info.tsp_order_index,
                         node_info.observation_score, node_info.cluster_distance);
            }
        }
        
        if (debug_enabled) {
            ROS_DEBUG("Added history odom node %d: pos(%.2f,%.2f,%.2f)", 
                     node_info.node_id, 
                     node_info.position.x(), node_info.position.y(), node_info.position.z());
        }
    }
    
    // 总结信息（总是输出）
    ROS_INFO("Enhanced extracted %d regions (%d with nodes), %d topo nodes, %d viewpoints", 
             (int)topo_graph_->regions_arr_.size(), regions_with_nodes, total_topo_nodes, total_viewpoints);
    
    if (debug_enabled) {
        ROS_INFO("🎯 Enhanced VIEWPOINT SUMMARY: Found %d viewpoints out of %d total nodes", total_viewpoints, total_topo_nodes);
    }
}

void TopoExtractorIntegrated::extractOdomNodeEnhanced(std::vector<NodeInfo>& nodes, int& node_counter) {
    // 静态调试标志
    static bool debug_enabled = []() {
        bool debug = false;
        ros::param::param("/exploration_node/topo_extraction/debug_output", debug, false);
        return debug;
    }();
    
    if (debug_enabled) {
        ROS_INFO("=== Enhanced Extracting Odom Node ===");
    }
    
    // 提取当前位置节点
    if (topo_graph_->odom_node_) {
        NodeInfo odom_node;
        odom_node.node_id = node_counter++;
        odom_node.position = topo_graph_->odom_node_->center_.cast<double>();
        odom_node.yaw = static_cast<double>(topo_graph_->odom_node_->yaw_);
        odom_node.is_viewpoint = topo_graph_->odom_node_->is_viewpoint_;  // 使用实际的viewpoint状态
        odom_node.is_current_odom = true;
        odom_node.is_history_odom = false;
        odom_node.region_id = -1; // 特殊标记为当前位置
        
        // 增强字段初始化
        odom_node.is_reachable = true; // 当前位置总是可达的
        odom_node.tsp_order_index = 0; // 在TSP中，索引0通常代表起始位置
        odom_node.reachable_distance = 0.0; // 到自身的距离为0
        
        // 收益信息字段 - 为当前odom节点也设置收益信息
        odom_node.observation_score = viewpoint_observation_score_map_.count(topo_graph_->odom_node_) > 0 ?
                                     viewpoint_observation_score_map_[topo_graph_->odom_node_] : 0.0;
        odom_node.cluster_distance = viewpoint_cluster_distance_map_.count(topo_graph_->odom_node_) > 0 ?
                                    viewpoint_cluster_distance_map_[topo_graph_->odom_node_] : -1.0;
        
        nodes.push_back(odom_node);
        
        if (topo_graph_->odom_node_->is_viewpoint_) {
            ROS_INFO("⭐ Current odom node %d is a VIEWPOINT at pos(%.2f,%.2f,%.2f) (TSP idx: %d, obs_score: %.1f, cluster_dist: %.2f)", 
                     odom_node.node_id,
                     odom_node.position.x(), odom_node.position.y(), odom_node.position.z(),
                     odom_node.tsp_order_index, odom_node.observation_score, odom_node.cluster_distance);
        } else if (debug_enabled) {
            ROS_INFO("Added current odom node %d: pos(%.2f,%.2f,%.2f), yaw=%.2f (not viewpoint, obs_score: %.1f, cluster_dist: %.2f)", 
                     odom_node.node_id,
                     odom_node.position.x(), odom_node.position.y(), odom_node.position.z(),
                     odom_node.yaw, odom_node.observation_score, odom_node.cluster_distance);
        }
    } else {
        ROS_WARN("No current odom node available");
    }
}

void TopoExtractorIntegrated::extractViewpointNodesEnhanced(std::vector<NodeInfo>& nodes, int& node_counter) {
    // 静态调试标志
    static bool debug_enabled = []() {
        bool debug = false;
        ros::param::param("/exploration_node/topo_extraction/debug_output", debug, false);
        return debug;
    }();
    
    if (debug_enabled) {
        ROS_INFO("=== Enhanced Extracting Viewpoint Nodes ===");
        ROS_INFO("Total viewpoints_update_region_arr_ size: %d", (int)topo_graph_->viewpoints_update_region_arr_.size());
    }
    
    // 提取视点节点 - 从viewpoints_update_region_arr_中的TopoNode提取
    int regions_with_viewpoints = 0;
    int total_viewpoint_nodes = 0;
    
    for (auto viewpoint_region : topo_graph_->viewpoints_update_region_arr_) {
        if (!viewpoint_region) continue;
        
        if (debug_enabled && !viewpoint_region->topo_nodes_.empty()) {
            ROS_INFO("Viewpoint region has %d topo nodes", (int)viewpoint_region->topo_nodes_.size());
        }
        regions_with_viewpoints++;
        
        // 从RegionNode中提取TopoNode
        for (auto topo_node : viewpoint_region->topo_nodes_) {
            if (!topo_node || !topo_node->is_viewpoint_) continue;
            
            total_viewpoint_nodes++;
            
            NodeInfo vp_node;
            vp_node.node_id = node_counter++;
            vp_node.position = topo_node->center_.cast<double>();
            vp_node.yaw = static_cast<double>(topo_node->yaw_);
            vp_node.is_viewpoint = true;
            vp_node.is_current_odom = false;
            vp_node.is_history_odom = false;
            vp_node.region_id = -2; // 特殊标记为视点
            
            // 增强字段初始化
            vp_node.is_reachable = viewpoint_reachability_map_.count(topo_node) > 0;
            vp_node.tsp_order_index = viewpoint_tsp_index_map_.count(topo_node) > 0 ? 
                                     viewpoint_tsp_index_map_[topo_node] : -1;
            vp_node.reachable_distance = viewpoint_distance_map_.count(topo_node) > 0 ?
                                        viewpoint_distance_map_[topo_node] : -1.0;
                                        
            // 收益信息字段 - 使用简化的收益信息
            vp_node.observation_score = viewpoint_observation_score_map_.count(topo_node) > 0 ?
                                       viewpoint_observation_score_map_[topo_node] : 0.0;
            vp_node.cluster_distance = viewpoint_cluster_distance_map_.count(topo_node) > 0 ?
                                      viewpoint_cluster_distance_map_[topo_node] : -1.0;
            
            if (debug_enabled) {
                ROS_INFO("VP node %d: obs_map=%s (%.1f), cluster_map=%s (%.2f)", 
                         vp_node.node_id,
                         viewpoint_observation_score_map_.count(topo_node) > 0 ? "FOUND" : "MISSING", vp_node.observation_score,
                         viewpoint_cluster_distance_map_.count(topo_node) > 0 ? "FOUND" : "MISSING", vp_node.cluster_distance);
            }
            
            nodes.push_back(vp_node);
            
            if (debug_enabled) {
                ROS_INFO("Added viewpoint %d: pos(%.2f,%.2f,%.2f), obs_score=%.1f, cluster_dist=%.2f", 
                         vp_node.node_id,
                         vp_node.position.x(), vp_node.position.y(), vp_node.position.z(),
                         vp_node.observation_score, vp_node.cluster_distance);
            }
        }
    }
    
    if (total_viewpoint_nodes > 0) {
        ROS_INFO("Enhanced extracted %d dedicated viewpoint nodes from %d regions", 
                 total_viewpoint_nodes, regions_with_viewpoints);
    }
}

} // namespace fast_planner
