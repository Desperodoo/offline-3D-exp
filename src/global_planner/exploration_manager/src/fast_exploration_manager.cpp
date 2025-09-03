/***
 * @Author: ning-zelin && zl.ning@qq.com
 * @Date: 2024-02-25 15:00:51
 * @LastEditTime: 2024-03-12 22:15:11
 * @Description:
 * @
 * @Copyright (c) 2024 by ning-zelin, All Rights Reserved.
 */

#include <boost/lexical_cast.hpp>
#include <epic_planner/expl_data.h>
#include <epic_planner/fast_exploration_manager.h>
#include <epic_planner/NeuralTSP.h>
#include <fstream>
#include <iostream>
#include <lkh_tsp_solver/lkh_interface.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <plan_manage/planner_manager.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <visualization_msgs/Marker.h>
using namespace std;
using namespace Eigen;

namespace fast_planner {
// SECTION interfaces for setup and query

FastExplorationManager::FastExplorationManager() {}

FastExplorationManager::~FastExplorationManager() {}

void FastExplorationManager::initialize(
    ros::NodeHandle &nh, FrontierManager::Ptr frt_manager,
    FastPlannerManager::Ptr planner_manager) {

  frontier_manager_ptr_ = frt_manager;
  planner_manager_ = planner_manager;

  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);
  ed_->next_goal_node_ = make_shared<TopoNode>();

  ep_->a_avg_ = tan(planner_manager_->gcopter_config_->maxTiltAngle) *
                planner_manager_->gcopter_config_->gravAcc;
  ep_->v_max_ = planner_manager_->gcopter_config_->maxVelMag;
  ep_->yaw_v_max_ = planner_manager_->gcopter_config_->yaw_max_vel;
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));
  nh.getParam("viewpoint_param/global_viewpoint_num",
              ep_->global_viewpoint_num_);
  nh.getParam("view_graph", ep_->view_graph_);
  nh.getParam("viewpoint_param/local_viewpoint_num", ep_->local_viewpoint_num_);
  nh.getParam("global_planning/w_vdir", ep_->w_vdir_);
  nh.getParam("global_planning/w_yawdir", ep_->w_yawdir_);
  Eigen::Vector3d origin, size;
  ofstream par_file(ep_->tsp_dir_ + "/single.par");
  par_file << "PROBLEM_FILE = " << ep_->tsp_dir_ << "/single.tsp\n";
  par_file << "GAIN23 = NO\n";
  par_file << "MOVE_TYPE = 2\n";
  par_file << "OUTPUT_TOUR_FILE =" << ep_->tsp_dir_ << "/single.txt\n";
  par_file << "RUNS = 10\n";
  
  // 初始化拓扑图提取器
  ROS_INFO("Initializing TopoExtractorIntegrated...");
  std::string topo_export_dir;
  bool debug_output = false;
  
  // 从ROS参数读取导出目录，默认使用EPIC项目下的topo_outputs目录
  nh.param("topo_extraction/export_dir", topo_export_dir, std::string("/home/amax/EPIC/topo_outputs"));
  nh.param("topo_extraction/debug_output", debug_output, false);
  
  topo_extractor_ = std::make_shared<TopoExtractorIntegrated>(
      planner_manager_->topo_graph_, topo_export_dir, true, 10.0);
  topo_extractor_->setDebugOutput(debug_output);
  ROS_INFO("✓ TopoExtractorIntegrated initialized - export dir: %s, debug: %s", 
           topo_export_dir.c_str(), debug_output ? "enabled" : "disabled");
  
  // 初始化神经网络TSP服务客户端
  nh.param("neural_tsp/use_neural_tsp", use_neural_tsp_, false);
  if (use_neural_tsp_) {
    neural_tsp_client_ = nh.serviceClient<epic_planner::NeuralTSP>("neural_tsp_solve");
    ROS_INFO("✓ Neural TSP client initialized, waiting for service...");
    if (!neural_tsp_client_.waitForExistence(ros::Duration(5.0))) {
      ROS_WARN("Neural TSP service not available, falling back to LKH solver");
      use_neural_tsp_ = false;
    } else {
      ROS_INFO("✓ Neural TSP service connected successfully");
    }
  } else {
    ROS_INFO("Using traditional LKH TSP solver");
  }
  
  ros::Duration(1.0).sleep();
}

void FastExplorationManager::goalCallback(
    const geometry_msgs::PoseStampedConstPtr &msg) {
  // 提取四元数
  double roll, pitch;
  tf::Quaternion quat;
  tf::quaternionMsgToTF(msg->pose.orientation, quat);

  // 将四元数转换为Euler角
  tf::Matrix3x3(quat).getRPY(roll, pitch, goal_yaw);
}

double FastExplorationManager::getPathCost(TopoNode::Ptr &n1,
                                           Eigen::Vector3d v1, float &yaw1,
                                           TopoNode::Ptr &n2, float &yaw2) {
  auto estimateCost = [&](TopoNode::Ptr &n1, Eigen::Vector3d v1, float &yaw1,
                          TopoNode::Ptr &n2, float &yaw2, int res,
                          vector<Eigen::Vector3f> &path) -> double {
    double len_cost, yaw_cost, dir_cost;
    len_cost = yaw_cost = dir_cost = 0.0;
    if (res == BubbleAstar::NO_PATH)
      return 2e3 + (n1->center_ - n2->center_)
                       .norm(); // 使用一个大的时间值表示无法到达
    if (res == BubbleAstar::START_FAIL || res == BubbleAstar::END_FAIL)
      return 2e3 +
             (n1->center_ - n2->center_).norm(); // 同上，用于不同的错误情况

    len_cost = 0.0;
    for (int i = 0; i < path.size() - 1; ++i)
      len_cost += ((path[i + 1] - path[i]).norm() +
                   0.5 * fabs(path[i + 1].z() - path[i].z()));
    len_cost /= (ep_->v_max_ / 2.0);

    // if (v1.norm() > 1e-3) {
    //   Eigen::Vector3f dir = n2->center_ - n1->center_;
    //   dir.normalize();
    //   Eigen::Vector3f v_dir = v1.normalized().cast<float>();
    //   float yaw1 = atan2(dir.y(), dir.x());
    //   float yaw2 = atan2(v_dir.y(), v_dir.x());
    //   float diff = yaw1 - yaw2;
    //   while (diff > M_PI)
    //     diff -= 2.0 * M_PI;
    //   while (diff < -M_PI)
    //     diff += 2.0 * M_PI;
    //   dir_cost = ep_->w_vdir_ * (fabs(diff) /
    //   planner_manager_->gcopter_config_->yaw_max_vel);
    // }

    // if (path.size() >= 2) {
    //   planner_manager_->calculateTimelb(path, yaw1, yaw2, yaw_cost);
    //   yaw_cost *= ep_->w_yawdir_;
    // }

    return len_cost + dir_cost;
    // return len_cost + dir_cost;
  };
  vector<Eigen::Vector3f> path;
  int res = planner_manager_->fast_searcher_->topoSearch(n1, n2, 1e-2, path);
  return estimateCost(n1, v1, yaw1, n2, yaw2, res, path);
}

double FastExplorationManager::getPathCostWithoutTopo(TopoNode::Ptr &n1,
                                                      Eigen::Vector3d v1,
                                                      float &yaw1,
                                                      TopoNode::Ptr &n2,
                                                      float &yaw2) {
  vector<Eigen::Vector3f> path;
  int res = planner_manager_->parallel_path_finder_->search(
      n1->center_, n2->center_, path, 1.0, false);
  if (res != ParallelBubbleAstar::REACH_END)
    return 2e3;
  double cost;
  planner_manager_->parallel_path_finder_->calculatePathCost(path, cost);
  return cost;
}

int FastExplorationManager::planGlobalPath(const Eigen::Vector3d &pos,
                                           const Eigen::Vector3d &vel) {
  bool bm_without_topo = false;
  auto estimiateVdirCost = [&](const TopoNode::Ptr &n1,
                               const Eigen::Vector3d &v1,
                               const TopoNode::Ptr &n2) -> double {
    Eigen::Vector3f dir = n2->center_ - n1->center_;
    dir.normalize();
    Eigen::Vector3f v_dir = v1.normalized().cast<float>();
    float yaw1 = atan2(dir.y(), dir.x());
    float yaw2 = atan2(v_dir.y(), v_dir.x());
    float diff = yaw1 - yaw2;
    while (diff > M_PI)
      diff -= 2.0 * M_PI;
    while (diff < -M_PI)
      diff += 2.0 * M_PI;
    return ep_->w_vdir_ *
           (fabs(diff) / planner_manager_->gcopter_config_->yaw_max_vel);
  };
  ros::Time start = ros::Time::now();
  vector<TopoNode::Ptr> viewpoints;
  vector<ViewpointBenefit> viewpoint_benefits; // 收集视点收益信息
  frontier_manager_ptr_->generateTSPViewpoints(
      planner_manager_->topo_graph_->odom_node_->center_, viewpoints, &viewpoint_benefits);

  if (viewpoints.empty()) {
    planner_manager_->graph_visualizer_->vizTour({}, VizColor::RED, "global");
    // 第一种情况：没有视点，保存拓扑图并更新探索统计
    if (topo_extractor_) {
      ROS_INFO("No viewpoints found - extracting final topo graph");
      topo_extractor_->extractTopoGraph("planGlobalPath_no_viewpoints");
    }
    // 更新探索统计 - 没有可达视点
    if (exploration_stats_) {
      exploration_stats_->updateViewpointCount(0);
    }
    return NO_FRONTIER;
  }

  ros::Time t1 = ros::Time::now();
  planner_manager_->topo_graph_->insertNodes(viewpoints, false);
  
  // 更新探索统计 - 估算探索面积
  if (exploration_stats_) {
    // 简单的探索面积估算：基于视点数量和覆盖范围
    double estimated_area = viewpoints.size() * 25.0; // 假设每个视点覆盖25平方米
    exploration_stats_->updateExploredArea(estimated_area);
    exploration_stats_->updateViewpointCount(viewpoints.size());
  }
  
  updateGoalNode();
  ros::Time t2 = ros::Time::now();
  cout << "insert viewpoint to graph time: " << (t2 - t1).toSec() * 1000
       << " ms" << endl;
  float curr_yaw = (float)planner_manager_->local_data_.curr_yaw_;
  vector<double> distance_odom2vp(viewpoints.size(), 0);
  vector<double> distance_lastgoal2vp(viewpoints.size(), 0);
  double dis2last_goal = 5e3;
  if (planner_manager_->lidar_map_interface_->getDisToOcc(
          ed_->next_goal_node_->center_) >
      planner_manager_->parallel_path_finder_->safe_distance_ + 0.1) {
    dis2last_goal = getPathCost(planner_manager_->topo_graph_->odom_node_,
                                Eigen::Vector3d::Zero(), curr_yaw,
                                ed_->next_goal_node_, curr_yaw);
  }
  static double last_frame_value = dis2last_goal;
  bool last_goal_reachable = dis2last_goal < 2e3;
  // last_goal_reachable = false;

  if (last_goal_reachable && (dis2last_goal < 1.5 * last_frame_value)) {
    last_frame_value = dis2last_goal;
  } else {
    last_goal_reachable = false;
  }

  ros::Time t_start_cvp_1 = ros::Time::now();
  omp_set_num_threads(4);
  // clang-format off
  #pragma omp parallel for
  // clang-format on
  for (int i = 0; i < viewpoints.size(); ++i) {
    if (last_goal_reachable) {
      distance_lastgoal2vp[i] =
          getPathCost(ed_->next_goal_node_, Eigen::Vector3d::Zero(),
                      viewpoints[i]->yaw_, viewpoints[i], viewpoints[i]->yaw_);
      distance_odom2vp[i] =
          getPathCost(planner_manager_->topo_graph_->odom_node_, vel, curr_yaw,
                      viewpoints[i], viewpoints[i]->yaw_);

    } else {
      distance_lastgoal2vp[i] =
          getPathCost(planner_manager_->topo_graph_->odom_node_, vel, curr_yaw,
                      viewpoints[i], viewpoints[i]->yaw_);
      distance_odom2vp[i] = distance_lastgoal2vp[i];
    }
  }
  ros::Time t_end_cvp_1 = ros::Time::now();
  if (bm_without_topo) {
    omp_set_num_threads(4);
    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (int i = 0; i < viewpoints.size(); ++i) {
      if (last_goal_reachable) {
        distance_lastgoal2vp[i] = getPathCostWithoutTopo(
            ed_->next_goal_node_, Eigen::Vector3d::Zero(), viewpoints[i]->yaw_,
            viewpoints[i], viewpoints[i]->yaw_);
        distance_odom2vp[i] = getPathCostWithoutTopo(
            planner_manager_->topo_graph_->odom_node_, vel, curr_yaw,
            viewpoints[i], viewpoints[i]->yaw_);

      } else {
        distance_lastgoal2vp[i] = getPathCostWithoutTopo(
            planner_manager_->topo_graph_->odom_node_, vel, curr_yaw,
            viewpoints[i], viewpoints[i]->yaw_);
        distance_odom2vp[i] = distance_lastgoal2vp[i];
      }
    }
    ros::Time t_end_cvp_2 = ros::Time::now();
    double cost_mat_with_topo = (t_end_cvp_1 - t_start_cvp_1).toSec() * 1000;
    double cost_mat_without_topo = (t_end_cvp_2 - t_end_cvp_1).toSec() * 1000;
    cout << "cost mat topo: " << cost_mat_with_topo << "ms" << endl;
    cout << "cost mat point cloud: " << cost_mat_without_topo << "ms" << endl;
  }

  vector<TopoNode::Ptr> viewpoint_reachable;
  vector<double> viewpoint_reachable_distance, viewpoint_reachable_distance2;
  for (int i = 0; i < distance_lastgoal2vp.size(); ++i) {
    if (distance_odom2vp[i] > 2e3)
      continue;
    if (last_goal_reachable) {
      viewpoint_reachable_distance.emplace_back(distance_lastgoal2vp[i]);

    } else {
      viewpoint_reachable_distance.emplace_back(distance_odom2vp[i]);
    }
    viewpoint_reachable_distance2.emplace_back(distance_odom2vp[i]);
    viewpoint_reachable.emplace_back(viewpoints[i]);
  }

  if (viewpoint_reachable.empty()) {
    planner_manager_->topo_graph_->removeNodes(viewpoints);
    planner_manager_->graph_visualizer_->vizTour({}, VizColor::RED, "global");
    // 第二种情况：有视点但都不可达，在移除视点之后保存拓扑图并更新探索统计
    if (topo_extractor_) {
      ROS_INFO("No reachable viewpoints - extracting topo graph after cleanup");
      topo_extractor_->extractTopoGraph("planGlobalPath_no_reachable");
    }
    // 更新探索统计 - 没有可达视点
    if (exploration_stats_) {
      exploration_stats_->updateViewpointCount(0);
    }
    return NO_FRONTIER;
  }

  if (viewpoint_reachable.size() == 1) {
    ed_->global_tour_.clear();
    ed_->global_tour_.emplace_back(pos.cast<float>());
    ed_->global_tour_.emplace_back(viewpoint_reachable.front()->center_);
    planner_manager_->local_data_.end_yaw_ = viewpoint_reachable.front()->yaw_;
    
    // 第三种情况：只有一个可达视点，在移除视点之前保存拓扑图并更新探索统计
    if (topo_extractor_) {
      ROS_INFO("Single reachable viewpoint - extracting topo graph before cleanup");
      topo_extractor_->extractTopoGraph("planGlobalPath_single_viewpoint");
    }
    // 更新探索统计 - 只有一个可达视点
    if (exploration_stats_) {
      exploration_stats_->updateViewpointCount(1);
    }
    
    planner_manager_->topo_graph_->removeNodes(viewpoints);
    return SUCCEED;
  }

  int dim = viewpoint_reachable.size() + 1;
  Eigen::MatrixXd mat;
  mat.resize(dim, dim);
  mat.setZero();
  for (int i = 1; i < dim; ++i) {
    mat(0, i) = viewpoint_reachable_distance[i - 1];
  }

  omp_set_num_threads(4);
  // clang-format off
  #pragma omp parallel for
  // clang-format on
  for (int i = 1; i < dim; i++) {
    for (int j = i + 1; j < dim; j++) {
      mat(i, j) = mat(j, i) = getPathCost(
          viewpoint_reachable[i - 1], Eigen::Vector3d(0, 0, 0),
          viewpoint_reachable[i - 1]->yaw_, viewpoint_reachable[j - 1],
          viewpoint_reachable[j - 1]->yaw_);
    }
  }
  // trick 往远走
  for (int i = 1; i < dim; ++i) {
    mat(i, 0) = 2e3 - viewpoint_reachable_distance2[i - 1] * 0.2;
  }
  for (int i = 0; i < dim; ++i) {
    for (int j = 1; j < dim; ++j) {
      for (int k = 1; k < dim; ++k) {
        if (mat(i, j) > mat(i, k) + mat(k, j)) {
          mat(i, j) = mat(i, k) + mat(k, j) + 1e-2;
        }
      }
    }
  }
  vector<int> indices;
  indices.reserve(dim);
  ros::Time start_tsp = ros::Time::now();
  cout << "calculate tsp cost matrix cost " << (start_tsp - t2).toSec() * 1000
       << "ms" << endl;
       
  // 尝试使用神经网络TSP求解
  bool neural_tsp_success = false;
  if (use_neural_tsp_ && viewpoint_reachable.size() > 1) {
    ROS_INFO("Attempting neural TSP with %zu viewpoints", viewpoint_reachable.size());
    int next_viewpoint_idx = solveNeuralTSP(viewpoint_reachable, pos);
    
    if (next_viewpoint_idx >= 0) {
      // 神经网络成功求解，构建简化的路径
      indices.clear();
      indices.push_back(0);  // 起始点
      indices.push_back(next_viewpoint_idx + 1);  // 选中的视点 (+1因为索引0是起始点)
      neural_tsp_success = true;
      ROS_INFO("✓ Neural TSP solved successfully, selected viewpoint %d", next_viewpoint_idx);
    } else {
      ROS_WARN("Neural TSP failed, falling back to LKH solver");
    }
  }
  
  // 如果神经网络求解失败，使用传统LKH求解器
  if (!neural_tsp_success) {
    solveLHK(mat, indices);
  }
  
  ros::Time end_tsp = ros::Time::now();
  cout << "tsp solver cost: " << (end_tsp - start_tsp).toSec() * 1000 << "ms"
       << endl;
  cout << "solver type: " << (neural_tsp_success ? "Neural" : "LKH") << endl;
  // if ((end_tsp - start_tsp).toSec() * 1000 > 100)
  //   exit(0);
  ed_->global_tour_.clear();
  for (auto &i : indices) {
    if (i == 0) {
      ed_->global_tour_.push_back(
          planner_manager_->topo_graph_->odom_node_->center_);
    } else {
      ed_->global_tour_.emplace_back(viewpoint_reachable[i - 1]->center_);
    }
  }
  if (!last_goal_reachable)
    last_frame_value = viewpoint_reachable_distance[indices[1]];

  // 更新拓扑提取器的viewpoint信息（简化版本）
  if (topo_extractor_) {
    ROS_INFO("Updating viewpoint info: %zu reachable, TSP order size: %zu, benefits: %zu", 
             viewpoint_reachable.size(), indices.size(), viewpoint_benefits.size());
    
    topo_extractor_->updateViewpointInfo(viewpoint_reachable, 
                                        viewpoint_reachable_distance2, 
                                        indices,
                                        viewpoint_benefits);
    
    // 重新提取拓扑图以包含更新的viewpoint信息和当前的探索统计
    topo_extractor_->extractTopoGraph("planGlobalPath_final");
    
    // 更新探索统计的可达视点数量
    if (exploration_stats_) {
      exploration_stats_->updateViewpointCount(viewpoint_reachable.size());
    }
  }

  ros::Time end = ros::Time::now();
  planner_manager_->topo_graph_->removeNodes(viewpoints);
  planner_manager_->graph_visualizer_->vizTour(ed_->global_tour_, VizColor::RED,
                                               "global");

  planner_manager_->local_data_.end_yaw_ =
      viewpoint_reachable[indices[1] - 1]->yaw_;
  updateGoalNode();
  return SUCCEED;
}

void FastExplorationManager::solveLHK(Eigen::MatrixXd &cost_mat,
                                      vector<int> &indices) {
  // Solve linear homogeneous kernel
  // Write params and cost matrix to problem file
  int dimension = cost_mat.rows();
  if (dimension < 3)
    return;
  ofstream prob_file(ep_->tsp_dir_ + "/single.tsp");
  // Problem specification part, follow the format of TSPLIB

  string prob_spec =
      "NAME : single\nTYPE : ATSP\nDIMENSION : " + to_string(dimension) +
      "\nEDGE_WEIGHT_TYPE : "
      "EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";

  // string prob_spec = "NAME : single\nTYPE : TSP\nDIMENSION : " +
  // to_string(dimension) +
  //     "\nEDGE_WEIGHT_TYPE : "
  //     "EXPLICIT\nEDGE_WEIGHT_FORMAT : LOWER_ROW\nEDGE_WEIGHT_SECTION\n";

  prob_file << prob_spec;
  // prob_file << "TYPE : TSP\n";
  // prob_file << "EDGE_WEIGHT_FORMAT : LOWER_ROW\n";
  // Problem data part
  const int scale = 100;

  // Use Asymmetric TSP
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = cost_mat(i, j) * scale;
      prob_file << int_cost << " ";
    }
    prob_file << "\n";
  }

  prob_file << "EOF";
  prob_file.close();

  // Call LKH TSP solver
  solveTSPLKH((ep_->tsp_dir_ + "/single.par").c_str());

  // Read optimal tour from the tour section of result file
  ifstream res_file(ep_->tsp_dir_ + "/single.txt");
  string res;
  while (getline(res_file, res)) {
    // Go to tour section
    if (res.compare("TOUR_SECTION") == 0)
      break;
  }

  // Read path for ATSP formulation
  while (getline(res_file, res)) {

    // Read indices of frontiers in optimal tour
    int id = stoi(res);
    if (id == -1)
      break;
    indices.push_back(id - 1); // Idx of solver-2 == Idx of frontier
  }

  res_file.close();
}

void FastExplorationManager::updateGoalNode() {
  if (ed_->global_tour_.empty())
    return;
  Eigen::Vector3f goal = ed_->global_tour_[1];

  struct PairPtrHash {
    std::size_t
    operator()(const std::pair<TopoNode::Ptr, TopoNode::Ptr> &p) const {
      return std::hash<TopoNode::Ptr>()(p.first) ^
             std::hash<TopoNode::Ptr>()(p.second);
    }
  };

  Eigen::Vector3i idx;
  planner_manager_->topo_graph_->getIndex(goal, idx);
  vector<TopoNode::Ptr> pre_nbrs;
  for (int i = -1; i <= 1; i++)
    for (int j = -1; j <= 1; j++)
      for (int k = -1; k <= 1; k++) {
        Eigen::Vector3i tmp_idx = idx;
        tmp_idx(0) = idx(0) + i;
        tmp_idx(1) = idx(1) + j;
        tmp_idx(2) = idx(2) + k;
        auto region = planner_manager_->topo_graph_->getRegionNode(tmp_idx);
        if (region) {
          for (auto &topo : region->topo_nodes_) {
            if (topo == ed_->next_goal_node_)
              continue;
            pre_nbrs.emplace_back(topo);
          }
        }
      }
  std::unordered_map<std::pair<TopoNode::Ptr, TopoNode::Ptr>,
                     vector<Eigen::Vector3f>, PairPtrHash>
      edge2insert;
  mutex edge2insert_mtx;
  omp_set_num_threads(4);
  // clang-format off
  #pragma omp parallel for
  // clang-format on
  for (auto &nbr : pre_nbrs) {
    vector<Eigen::Vector3f> path;
    int res = planner_manager_->topo_graph_->parallel_bubble_astar_->search(
        goal, nbr->center_, path, 1e-3);
    if (res == ParallelBubbleAstar::REACH_END &&
        planner_manager_->topo_graph_->parallel_bubble_astar_
            ->collisionCheck_shortenPath(path)) {
      edge2insert_mtx.lock();
      edge2insert.insert({std::make_pair(ed_->next_goal_node_, nbr), path});
      edge2insert_mtx.unlock();
    }
  }
  // 更新goal节点
  ed_->next_goal_node_->center_ = goal;
  ed_->next_goal_node_->is_viewpoint_ = true;
  if (edge2insert.size() > 0) {
    planner_manager_->topo_graph_->removeNode(ed_->next_goal_node_);
    for (auto &edge : edge2insert) {
      ed_->next_goal_node_->neighbors_.insert(edge.first.second);
      ed_->next_goal_node_->paths_.insert({edge.first.second, edge.second});
      double cost;
      planner_manager_->topo_graph_->parallel_bubble_astar_->calculatePathCost(
          edge.second, cost);
      ed_->next_goal_node_->weight_[edge.first.second] = cost;
      auto nbr = edge.first.second;
      nbr->neighbors_.insert(ed_->next_goal_node_);
      nbr->weight_[ed_->next_goal_node_] = cost;
      vector<Eigen::Vector3f> path = edge.second;
      std::reverse(path.begin(), path.end());
      nbr->paths_[ed_->next_goal_node_] = path;
    }
  }
}

int FastExplorationManager::solveNeuralTSP(const vector<TopoNode::Ptr> &viewpoint_reachable, 
                                          const Eigen::Vector3d &current_pos) {
  /**
   * 使用神经网络求解TSP问题，返回下一个目标视点的索引
   * 传递完整的拓扑图信息给neural_tsp_server
   * 
   * @param viewpoint_reachable: 可达视点列表
   * @param current_pos: 当前位置
   * @return: 下一个目标视点在viewpoint_reachable中的索引，-1表示失败
   */
  
  if (!use_neural_tsp_ || viewpoint_reachable.empty()) {
    return -1;
  }
  
  try {
    // 构建服务请求
    epic_planner::NeuralTSP srv;
    
    // 设置当前位置
    srv.request.current_position.x = current_pos.x();
    srv.request.current_position.y = current_pos.y();
    srv.request.current_position.z = current_pos.z();
    
    // 获取完整拓扑图信息
    auto topo_graph = planner_manager_->topo_graph_;
    
    // 从区域映射中收集所有拓扑节点
    vector<TopoNode::Ptr> all_topo_nodes;
    
    // 遍历所有区域来收集节点
    for (const auto& region_pair : topo_graph->reg_map_idx2ptr_) {
      const auto& region_node = region_pair.second;  // RegionNode::Ptr
      for (const auto& topo_node : region_node->topo_nodes_) {
        all_topo_nodes.push_back(topo_node);
      }
    }
    
    // 如果没有从区域中找到节点，添加odom节点作为backup
    if (all_topo_nodes.empty() && topo_graph->odom_node_) {
      all_topo_nodes.push_back(topo_graph->odom_node_);
    }
    
    ROS_INFO("Constructing neural TSP request with %zu nodes, %zu viewpoints", 
             all_topo_nodes.size(), viewpoint_reachable.size());
    
    // 构建拓扑图节点信息 - 与epic3d_data_processor.py格式完全对齐
    srv.request.node_positions.clear();
    srv.request.node_yaws.clear();
    srv.request.node_is_viewpoint.clear();
    srv.request.node_is_current.clear();
    srv.request.node_is_history.clear();
    srv.request.node_region_ids.clear();          // 新增：region_id字段
    srv.request.node_is_reachable.clear();
    srv.request.node_tsp_order.clear();          // 修正：与srv定义对齐
    srv.request.node_distances.clear();
    srv.request.node_observation_scores.clear();
    srv.request.node_cluster_distances.clear();
    
    // 创建节点索引映射
    std::unordered_map<TopoNode::Ptr, int> node_to_index;
    std::unordered_map<TopoNode::Ptr, int> node_to_region_id;
    
    // 构建region_id映射
    for (const auto& region_pair : topo_graph->reg_map_idx2ptr_) {
      const Eigen::Vector3i& region_idx = region_pair.first;  // Vector3i类型的region index
      // 使用region index的哈希值作为region_id
      int region_id = std::abs(region_idx.x() * 1000 + region_idx.y() * 100 + region_idx.z());
      const auto& region_node = region_pair.second;
      for (const auto& topo_node : region_node->topo_nodes_) {
        node_to_region_id[topo_node] = region_id + 1000;  // 与txt文件格式对齐
      }
    }
    
    for (size_t i = 0; i < all_topo_nodes.size(); ++i) {
      auto node = all_topo_nodes[i];
      node_to_index[node] = i;
      
      // 节点位置 (字段1-3: x, y, z)
      geometry_msgs::Point pos;
      pos.x = node->center_.x();
      pos.y = node->center_.y(); 
      pos.z = node->center_.z();
      srv.request.node_positions.push_back(pos);
      
      // 航向角 (字段4: yaw)
      srv.request.node_yaws.push_back(node->yaw_);
      
      // 是否为视点 (字段5: is_viewpoint)
      bool is_viewpoint = std::find(viewpoint_reachable.begin(), viewpoint_reachable.end(), node) != viewpoint_reachable.end();
      srv.request.node_is_viewpoint.push_back(is_viewpoint || node->is_viewpoint_);
      
      // 是否为当前位置 (字段6: is_current)
      bool is_current = (node == topo_graph->odom_node_) || 
                       ((node->center_ - current_pos.cast<float>()).norm() < 0.5f);
      srv.request.node_is_current.push_back(is_current);
      
      // 历史状态 (字段7: is_history)
      bool is_history = node->is_history_odom_node_;
      srv.request.node_is_history.push_back(is_history);
      
      // Region ID (字段8: region_id)
      auto region_it = node_to_region_id.find(node);
      int region_id = (region_it != node_to_region_id.end()) ? region_it->second : 0;
      srv.request.node_region_ids.push_back(region_id);
      
      // 可达性 (字段9: is_reachable)
      srv.request.node_is_reachable.push_back(true);
      
      // TSP顺序索引 (字段10: tsp_order_index)
      // 对于当前推理阶段，大部分节点设为-1（未分配）
      // 只有当前节点设为0，视点节点可以设为其他值或-1
      int tsp_order = -1;
      if (is_current) {
        tsp_order = 0;  // 当前位置为起始点
      }
      srv.request.node_tsp_order.push_back(tsp_order);
      
      // 距离计算 (字段11: distance)
      float distance = (node->center_ - current_pos.cast<float>()).norm();
      srv.request.node_distances.push_back(distance);
      
      // 观测得分 (字段12: observation_score)
      // 基于是否为视点和距离的启发式计算
      float obs_score = 0.0f;
      if (is_viewpoint) {
        obs_score = std::max(0.0f, 10.0f - distance * 0.1f);
      }
      srv.request.node_observation_scores.push_back(obs_score);
      
      // 聚类距离 (字段13: cluster_distance)
      // 对于推理阶段，可以使用启发式值或-1.0表示未计算
      float cluster_dist = is_viewpoint ? distance * 0.8f : -1.0f;
      srv.request.node_cluster_distances.push_back(cluster_dist);
    }
    
    // 构建边信息
    srv.request.edge_from_nodes.clear();
    srv.request.edge_to_nodes.clear();
    srv.request.edge_weights.clear();
    srv.request.edge_is_reachable.clear();
    
    for (size_t i = 0; i < all_topo_nodes.size(); ++i) {
      auto node = all_topo_nodes[i];
      for (const auto& neighbor : node->neighbors_) {
        auto it = node_to_index.find(neighbor);
        if (it != node_to_index.end()) {
          srv.request.edge_from_nodes.push_back(i);
          srv.request.edge_to_nodes.push_back(it->second);
          
          // 权重信息
          auto weight_it = node->weight_.find(neighbor);
          float weight = (weight_it != node->weight_.end()) ? weight_it->second : 1.0f;
          srv.request.edge_weights.push_back(weight);
          
          // 可达性（检查unreachable列表）
          bool is_reachable = node->unreachable_nbrs_.find(neighbor) == node->unreachable_nbrs_.end();
          srv.request.edge_is_reachable.push_back(is_reachable);
        }
      }
    }
    
    // 设置视点信息
    srv.request.viewpoints.clear();
    srv.request.viewpoint_indices.clear();
    srv.request.viewpoint_distances.clear();
    
    for (size_t i = 0; i < viewpoint_reachable.size(); ++i) {
      auto vp = viewpoint_reachable[i];
      
      // 视点位置
      geometry_msgs::Point vp_pos;
      vp_pos.x = vp->center_.x();
      vp_pos.y = vp->center_.y();
      vp_pos.z = vp->center_.z();
      srv.request.viewpoints.push_back(vp_pos);
      
      // 视点在拓扑图中的索引
      auto it = node_to_index.find(vp);
      int topo_index = (it != node_to_index.end()) ? it->second : i;  // fallback to local index
      srv.request.viewpoint_indices.push_back(topo_index);
      
      // 视点距离
      float vp_distance = (vp->center_ - current_pos.cast<float>()).norm();
      srv.request.viewpoint_distances.push_back(vp_distance);
    }
    
    // 探索统计信息
    srv.request.viewpoints_found = viewpoint_reachable.size();
    srv.request.viewpoints_reachable = viewpoint_reachable.size();
    srv.request.viewpoints_visited = 0;  // 简化处理
    srv.request.exploration_area = 0.0f;  // 可以从exploration_stats_获取
    srv.request.exploration_efficiency = 1.0f;  // 简化处理
    
    // 调用神经网络服务
    ros::Time start_time = ros::Time::now();
    ROS_INFO("Calling neural TSP service with complete topo graph: %zu nodes, %zu edges, %zu viewpoints",
             srv.request.node_positions.size(), srv.request.edge_from_nodes.size(), viewpoint_reachable.size());
             
    if (neural_tsp_client_.call(srv)) {
      ros::Time end_time = ros::Time::now();
      
      if (srv.response.success) {
        int next_idx = srv.response.next_viewpoint_index;
        
        // 验证索引有效性
        if (next_idx >= 0 && next_idx < viewpoint_reachable.size()) {
          ROS_INFO("Neural TSP solved: next viewpoint index = %d, time = %.2f ms, confidence = %.2f", 
                   next_idx, (end_time - start_time).toSec() * 1000, srv.response.confidence_score);
          return next_idx;
        } else {
          ROS_WARN("Neural TSP returned invalid index: %d (valid range: 0-%zu)", 
                   next_idx, viewpoint_reachable.size()-1);
          return -1;
        }
      } else {
        ROS_WARN("Neural TSP failed: %s", srv.response.message.c_str());
        return -1;
      }
    } else {
      ROS_ERROR("Failed to call neural TSP service");
      return -1;
    }
    
  } catch (const std::exception& e) {
    ROS_ERROR("Exception in neural TSP solving: %s", e.what());
    return -1;
  }
}
} // namespace fast_planner