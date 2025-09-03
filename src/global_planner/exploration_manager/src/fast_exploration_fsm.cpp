/***
 * @Author: ning-zelin && zl.ning@qq.com
 * @Date: 2024-02-29 16:54:46
 * @LastEditTime: 2024-03-11 13:22:44
 * @Description:
 * @
 * @Copyright (c) 2024 by ning-zelin, All Rights Reserved.
 */

#include <epic_planner/expl_data.h>
#include <epic_planner/fast_exploration_fsm.h>
#include <epic_planner/fast_exploration_manager.h>
#include <plan_manage/planner_manager.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <traj_utils/planning_visualization.h>
using Eigen::Vector3d;
using Eigen::Vector4d;
bool debug_planner;
typedef visualization_msgs::Marker Marker;
typedef visualization_msgs::MarkerArray MarkerArray;

void FastExplorationFSM::FSMCallback(const ros::TimerEvent &e) {
  pubState();
  switch (state_) {
  case INIT: {
    if (!fd_->have_odom_) {
      ROS_WARN_THROTTLE(1.0, "no odom.");
      return;
    }
    transitState(WAIT_TRIGGER, "FSM");
    break;
  }

  case WAIT_TRIGGER: {
    // 检查自动开始条件
    if (fp_->enable_auto_start_ && !fd_->auto_start_triggered_ && checkAutoStartCondition()) {
      fd_->trigger_ = true;
      fd_->auto_start_triggered_ = true;
      fd_->exploration_start_time_ = ros::Time::now();  // 记录探索开始时间
      ROS_INFO("\033[32m[Auto Start] Exploration started automatically! Condition: %s, Delay: %.1fs\033[0m", 
               fp_->auto_start_condition_.c_str(), fp_->auto_start_delay_);
      transitState(PLAN_TRAJ, "auto_start");
      break;
    }
    
    if (fp_->enable_auto_start_) {
      double elapsed = (ros::Time::now() - fd_->system_start_time_).toSec();
      bool odom_ready = fd_->have_odom_;
      bool topo_ready = !planner_manager_->topo_graph_->odom_node_->neighbors_.empty();
      
      ROS_WARN_THROTTLE(2.0, "[Auto Start] Waiting for condition '%s' (%.1fs/%.1fs) - Odom: %s, Topo: %s", 
                        fp_->auto_start_condition_.c_str(), elapsed, fp_->auto_start_delay_, 
                        odom_ready ? "✓" : "✗", topo_ready ? "✓" : "✗");
    } else {
      ROS_WARN_THROTTLE(1.0, "wait for trigger.");
    }
    break;
  }

  case FINISH: {
    // stopTraj();
    double collision_time = 0.0;
    bool safe = planner_manager_->checkTrajCollision(collision_time);
    if (!safe) {
      stopTraj();
    }
    
    // 检查安全退出条件
    if (fp_->enable_safe_exit_ && !fd_->safe_exit_initiated_ && !fd_->finish_state_enter_time_.isZero()) {
      double finish_elapsed = (ros::Time::now() - fd_->finish_state_enter_time_).toSec();
      if (finish_elapsed >= fp_->safe_exit_delay_) {
        fd_->safe_exit_initiated_ = true;
        ROS_WARN("\033[35m[Safe Exit] Initiating safe shutdown after %.1fs in FINISH state\033[0m", 
                 finish_elapsed);
        
        // 停止所有轨迹
        stopTraj();
        
        // 发布最终状态
        std_msgs::String final_status;
        if (fd_->exploration_timeout_) {
          final_status.data = "EXPLORATION_TERMINATED_TIMEOUT";
        } else if (fd_->exploration_completed_) {
          final_status.data = "EXPLORATION_TERMINATED_COMPLETED";  
        } else {
          final_status.data = "EXPLORATION_TERMINATED_NO_FRONTIER";
        }
        auto_stop_status_pub_.publish(final_status);
        
        // 停止所有定时器，避免线程冲突
        exec_timer_.stop();
        
        // 延迟一点时间确保定时器停止
        ros::Duration(0.5).sleep();
        
        // 安全关闭ROS节点
        ROS_INFO("\033[35m[Safe Exit] Shutting down exploration node safely...\033[0m");
        
        // 使用更温和的退出方式
        ros::requestShutdown();
        return;
      } else {
        ROS_INFO_THROTTLE(2.0, "\033[36m[Safe Exit] Will exit in %.1fs\033[0m", 
                          fp_->safe_exit_delay_ - finish_elapsed);
      }
    }
    
    // 显示不同的完成原因
    if (fd_->exploration_timeout_) {
      ROS_WARN_THROTTLE(1.0, "\033[33mExploration FINISHED: TIMEOUT\033[0m");
    } else if (fd_->exploration_completed_) {
      ROS_WARN_THROTTLE(1.0, "\033[33mExploration FINISHED: COMPLETED\033[0m");
    } else {
      ROS_WARN_THROTTLE(1.0, "\033[33mExploration FINISHED: NO_FRONTIER\033[0m");
    }
    break;
  }

  case PLAN_TRAJ: {
    if (!fd_->trigger_)
      return;
    if (planner_manager_->topo_graph_->odom_node_->neighbors_.empty())
      return;
    
    // 检查自动停止条件
    if (fp_->enable_auto_stop_ && checkAutoStopCondition()) {
      if (fd_->exploration_timeout_) {
        transitState(FINISH, "exploration_timeout");
      } else if (fd_->exploration_completed_) {
        transitState(FINISH, "exploration_completed");
      }
      return;
    }
      
    ros::Time start = ros::Time::now();
    // 要报min-step的case
    LocalTrajData *info = &planner_manager_->local_data_;
    double t_cur = (ros::Time::now() - info->start_time_).toSec();
    // double time_to_end = info->duration_ - t_cur;  // 暂时未使用
    if (expl_manager_->ed_->global_tour_.size() == 2) {
      Eigen::Vector3f goal = expl_manager_->ed_->global_tour_[1];
      if ((goal - fd_->odom_pos_).norm() < 1e-1) {

        // transitState(FINISH, "fsm");
        // return;

        // 检查是否使用Neural TSP模式
        if (expl_manager_->use_neural_tsp_) {
          // Neural TSP模式：到达当前视点不意味着探索完成
          // 触发重新规划，寻找下一个目标
          ROS_INFO("Neural TSP: Reached current viewpoint, triggering replanning");
          // 清空当前tour，让系统重新规划下一个目标
          expl_manager_->ed_->global_tour_.clear();
          // 保持在PLAN_TRAJ状态，不转换到FINISH
          return;
        } else {
          // LKH模式：原有逻辑，到达终点表示探索完成
          ROS_INFO("LKH mode: Reached final waypoint, exploration completed");
          transitState(FINISH, "fsm - LKH mode completed");
          return;
        }

      }
    }
    ros::Time tplan = ros::Time::now();
    exec_timer_.stop();
    int res = callExplorationPlanner();
    exec_timer_.start();
    ROS_INFO("\033[31m call planner \033[0m: %.3f",
             (ros::Time::now() - tplan).toSec() * 1000.0);

    if (res == SUCCEED) {
      poly_yaw_traj_pub_.publish(fd_->newest_yaw_traj_);
      poly_traj_pub_.publish(fd_->newest_traj_);
      fd_->static_state_ = false;
      if (fd_->use_bubble_a_star_) {
        transitState(EXEC_TRAJ,
                     "ParallelBubbleAstar plan success: new traj pub");
      } else {
        transitState(EXEC_TRAJ, "plan success: new traj pub");
      }
      fd_->use_bubble_a_star_ = false;
      fd_->half_resolution = false;

    } else if (res == NO_FRONTIER) {
      // NO_FRONTIER直接表示探索完成
      ROS_INFO("[Auto Stop] NO_FRONTIER detected - exploration completed");
      
      // if (planner_manager_->topo_graph_->global_view_points_.empty())
      transitState(FINISH, "PLAN_TRAJ: no frontier");
      fd_->static_state_ = true;
    } else if (res == FAIL) {
      // Still in PLAN_TRAJ state, keep replanning
      stopTraj();
      transitState(PLAN_TRAJ, "PLAN_TRAJ: plan failed", true);

    } else if (res == START_FAIL) {
      transitState(CAUTION, "PLAN_TRAJ: start failed", true);
    } else {
      cout << "330?" << endl;
    }
    break;
  }

  case EXEC_TRAJ: {
    // 检查自动停止条件
    if (fp_->enable_auto_stop_ && checkAutoStopCondition()) {
      if (fd_->exploration_timeout_) {
        stopTraj();
        transitState(FINISH, "exploration_timeout");
        return;
      } else if (fd_->exploration_completed_) {
        stopTraj();
        transitState(FINISH, "exploration_completed");
        return;
      }
    }
    
    // collision check
    double collision_time;
    bool safe = planner_manager_->checkTrajCollision(collision_time);
    if (!safe) {
      transitState(
          PLAN_TRAJ,
          "safetyCallback: not safe, time:" + to_string(collision_time), true);
      if (collision_time < fp_->replan_time_ + 0.2)
        stopTraj();
    } else if (!planner_manager_->checkTrajVelocity()) {
      transitState(PLAN_TRAJ, "velocity too fast", true);
    }

    break;
  }

  case CAUTION: {
    stopTraj();
    exec_timer_.stop();
    bool success = planner_manager_->flyToSafeRegion(fd_->static_state_);
    if (success) {
      traj_utils::PolyTraj poly_traj_msg;
      auto info = &planner_manager_->local_data_;
      planner_manager_->polyTraj2ROSMsg(poly_traj_msg, info->start_time_);
      fd_->newest_traj_ = poly_traj_msg;
      poly_traj_pub_.publish(fd_->newest_traj_);
      ros::Duration(0.2).sleep();
    }
    exec_timer_.start();
    double dis2occ =
        planner_manager_->lidar_map_interface_->getDisToOcc(fd_->odom_pos_);
    if (dis2occ > planner_manager_->gcopter_config_->dilateRadiusSoft)
      transitState(PLAN_TRAJ, "safe now");
    break;
  }
  case LAND: {
    stopTraj();
    exec_timer_.stop();
    global_path_update_timer_.stop();
    // 没电了！！再飞就会炸鸡，降落！！！
    while (1) {
      quadrotor_msgs::TakeoffLand land_msg;
      land_msg.takeoff_land_cmd = land_msg.LAND;
      land_pub_.publish(land_msg);
      ros::Duration(0.2).sleep();
      ROS_WARN_THROTTLE(1.0, "NO POWER. LAND!!");
    }

    break;
  }
  }
}

void FastExplorationFSM::init(ros::NodeHandle &nh,
                              FastExplorationManager::Ptr &explorer) {
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /*  Fsm param  */
  nh.param("fsm/thresh_replan", fp_->replan_thresh_, -1.0);
  nh.param("fsm/replan_time", fp_->replan_time_, -1.0);
  nh.param("bubble_astar/resolution_astar", fp_->bubble_a_star_resolution, 0.1);
  nh.param("fsm/debug_planner", debug_planner, false);
  nh.param("fsm/emergency_replan_control_error",
           fp_->emergency_replan_control_error, 0.3);
  nh.param("fsm/replan_time_after_traj_start",
           fp_->replan_time_after_traj_start_, 0.5);
  nh.param("fsm/replan_time_before_traj_end", fp_->replan_time_before_traj_end_,
           0.5);
           
  // 自动开始探索参数
  nh.param("fsm/enable_auto_start", fp_->enable_auto_start_, false);
  nh.param("fsm/auto_start_delay", fp_->auto_start_delay_, 5.0);
  nh.param("fsm/auto_start_condition", fp_->auto_start_condition_, string("odom_ready"));
  
  // 自动停止探索参数
  nh.param("fsm/enable_auto_stop", fp_->enable_auto_stop_, false);
  nh.param("fsm/max_exploration_time", fp_->max_exploration_time_, 1800.0);  // 30分钟
  nh.param("fsm/completion_check_interval", fp_->completion_check_interval_, 5.0);  // 5秒检查一次
  
  // 安全退出参数
  nh.param("fsm/enable_safe_exit", fp_->enable_safe_exit_, false);
  nh.param("fsm/safe_exit_delay", fp_->safe_exit_delay_, 10.0);  // FINISH状态10秒后退出
  
  /* Initialize main modules */
  // expl_manager_.reset(new FastExplorationManager);
  // expl_manager_->initialize(nh);
  expl_manager_ = explorer;
  planner_manager_ = expl_manager_->planner_manager_;

  state_ = EXPL_STATE::INIT;
  fd_->have_odom_ = false;
  fd_->state_str_ = {"INIT",      "WAIT_TRIGGER", "PLAN_TRAJ", "CAUTION",
                     "EXEC_TRAJ", "FINISH",       "LAND"};
  fd_->static_state_ = true;
  fd_->trigger_ = false;
  fd_->use_bubble_a_star_ = false;
  
  // 初始化自动开始相关状态
  fd_->system_start_time_ = ros::Time::now();
  fd_->auto_start_triggered_ = false;
  
  // 初始化自动停止相关状态
  fd_->exploration_start_time_ = ros::Time(0);
  fd_->last_completion_check_time_ = ros::Time::now();
  fd_->exploration_timeout_ = false;
  fd_->exploration_completed_ = false;
  
  // 初始化安全退出相关状态
  fd_->finish_state_enter_time_ = ros::Time(0);
  fd_->safe_exit_initiated_ = false;
  battary_sub_ =
      nh.subscribe("/mavros/battery", 10, &FastExplorationFSM::battaryCallback,
                   this, ros::TransportHints().tcpNoDelay());

  /* Ros sub, pub and timer */
  // if (debug_planner) {
  //   exec_timer_ = nh.createTimer(ros::Duration(0.01),
  //   &FastExplorationFSM::PlannerDebugFSMCallback, this);
  // } else {
  exec_timer_ = nh.createTimer(ros::Duration(0.01),
                               &FastExplorationFSM::FSMCallback, this);
  // }
  global_path_update_timer_ = nh.createTimer(
      ros::Duration(0.2), &FastExplorationFSM::globalPathUpdateCallback, this);
  trigger_sub_ = nh.subscribe("/waypoint_generator/waypoints", 1,
                              &FastExplorationFSM::triggerCallback, this);
  replan_pub_ = nh.advertise<std_msgs::Empty>("/planning/replan", 10);

  heartbeat_pub_ = nh.advertise<std_msgs::Empty>("/planning/heartbeat", 10);
  land_pub_ =
      nh.advertise<quadrotor_msgs::TakeoffLand>("/px4ctrl/takeoff_land", 10);

  poly_traj_pub_ =
      nh.advertise<traj_utils::PolyTraj>("/planning/trajectory", 10);
  poly_yaw_traj_pub_ =
      nh.advertise<traj_utils::PolyTraj>("/planning/yaw_trajectory", 10);
  time_cost_pub_ = nh.advertise<std_msgs::Float32>("/time_cost", 10);
  static_pub_ = nh.advertise<std_msgs::Bool>("/planning/static", 10);
  state_pub_ = nh.advertise<visualization_msgs::Marker>("/planning/state", 10);
  auto_start_status_pub_ = nh.advertise<std_msgs::String>("/exploration/auto_start_status", 10);
  auto_stop_status_pub_ = nh.advertise<std_msgs::String>("/exploration/auto_stop_status", 10);

  string odom_topic, cloud_topic;
  nh.getParam("odometry_topic", odom_topic);
  nh.getParam("cloud_topic", cloud_topic);
  cloud_sub_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(
      nh, cloud_topic, 1));
  odom_sub_.reset(
      new message_filters::Subscriber<nav_msgs::Odometry>(nh, odom_topic, 5));
  sync_cloud_odom_.reset(new message_filters::Synchronizer<SyncPolicyCloudOdom>(
      SyncPolicyCloudOdom(10), *cloud_sub_, *odom_sub_));
  sync_cloud_odom_->registerCallback(
      boost::bind(&FastExplorationFSM::CloudOdomCallback, this, _1, _2));
}

void FastExplorationFSM::battaryCallback(
    const sensor_msgs::BatteryStateConstPtr &msg) {
  // if(msg->voltage < 21.0){
  //   transitState(LAND, "battary low");
  // }
}

void FastExplorationFSM::updateTopoAndGlobalPath() {
  if (!(state_ == WAIT_TRIGGER || state_ == PLAN_TRAJ || state_ == EXEC_TRAJ ||
        state_ == FINISH)) {
    global_path_update_timer_.stop();
    // expl_manager_->frontier_manager_ptr_->viz_pocc();
    expl_manager_->frontier_manager_ptr_->visfrtcluster();
    global_path_update_timer_.start();
    return;
  }
  static int cnt = 0;
  cnt++;

  global_path_update_timer_.stop();
  ros::Time t2 = ros::Time::now();
  planner_manager_->topo_graph_->getRegionsToUpdate();
  // cout << "getRegionsToUpdate time cost:" << (ros::Time::now() - t2).toSec()
  // * 1000 << "ms" << endl;
  planner_manager_->topo_graph_->updateSkeleton();

  ros::Time t3 = ros::Time::now();
  planner_manager_->topo_graph_->updateOdomNode(fd_->odom_pos_, fd_->odom_yaw_);
  planner_manager_->topo_graph_->updateHistoricalOdoms();

  if (planner_manager_->topo_graph_->odom_node_->neighbors_.empty()) {
    double time;
    if (planner_manager_->local_data_.traj_id_ > 1) {
      bool safe = planner_manager_->checkTrajCollision(time);
      if (!safe) {
        transitState(CAUTION, "odom_node no nbrs");
      } else {
        global_path_update_timer_.start();

        return;
      }
    } else {
      transitState(CAUTION, "odom_node no nbrs");
    }
    global_path_update_timer_.start();
    return;
  }
  if (planner_manager_->local_data_.traj_id_ > 1) {

    double curr_time =
        (ros::Time::now() - planner_manager_->local_data_.start_time_).toSec();
    double time;
    bool safe = planner_manager_->checkTrajCollision(time);
    double total_time = planner_manager_->local_data_.duration_;
    double time2end = total_time - curr_time;

    if (safe && curr_time < fp_->replan_time_after_traj_start_ &&
        time2end > fp_->replan_time_before_traj_end_) {
      global_path_update_timer_.start();
      return;
    }
  }
  cout << endl << endl;
  cout << "\033[1;33m------------- <" << cnt
       << "> Plan Global Path start---------------" << "\033[0m" << endl;
  planner_manager_->topo_graph_->log << "<" << cnt << ">" << endl;
  ros::Time t4 = ros::Time::now();
  // cout << "updateSkeleton time cost:" << (t3 - t2).toSec() * 1000 << "ms" <<
  // endl; if( (t3 - t1).toSec() * 1000 > 100){
  //   ROS_ERROR("time too long");
  //   exit(0);
  // }
  ROS_INFO("update topo skeleton cost: %fms, update odom vertex cost:%fms ",
           (t3 - t2).toSec() * 1000, (t4 - t3).toSec() * 1000);
  Eigen::Vector3d vel = fd_->odom_vel_.cast<double>();
  Eigen::Vector3d odom = fd_->odom_pos_.cast<double>();
  int res = expl_manager_->planGlobalPath(odom, vel);
  ros::Time t5 = ros::Time::now();

  cout << "\033[1;33m-------------Plan Global Path end-----------------"
       << "\033[0m" << endl
       << endl;

  planner_manager_->graph_visualizer_->vizBox(planner_manager_->topo_graph_);
  if(expl_manager_->ep_->view_graph_)
    planner_manager_->graph_visualizer_->vizGraph(planner_manager_->topo_graph_);
  std_msgs::Float32 time_cost;
  double time_cost_now = (t5 - t2).toSec() * 1000;
  time_cost.data = time_cost_now;
  time_cost_pub_.publish(time_cost);

  cout << "total time cost: " << time_cost_now << "ms" << endl;
  if (res == NO_FRONTIER && state_ != WAIT_TRIGGER) {
    transitState(FINISH, "planGlobalPath: no frontier");
  } else if (res == SUCCEED && state_ != WAIT_TRIGGER) {
    transitState(PLAN_TRAJ, "planGlobalPath: succeed");
  }

  expl_manager_->frontier_manager_ptr_->viz_pocc();
  expl_manager_->frontier_manager_ptr_->visfrtcluster();
  static ros::Time t_p = ros::Time::now();
  if ((ros::Time::now() - t_p).toSec() > 5.0) {
    expl_manager_->frontier_manager_ptr_->printMemoryCost();
    t_p = ros::Time::now();
  }
  global_path_update_timer_.start();
  cout << "viz&&print cost:" << (ros::Time::now() - t5).toSec() * 1000 << "ms"
       << endl;
}

void FastExplorationFSM::globalPathUpdateCallback(const ros::TimerEvent &e) {
  updateTopoAndGlobalPath();
}
