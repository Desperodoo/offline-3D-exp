/**
 * @file exploration_statistics.cpp
 * @brief 探索统计模块实现
 */

#include <epic_planner/exploration_statistics.h>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace fast_planner {

ExplorationStatistics::ExplorationStatistics() 
    : initialized_(false), min_distance_threshold_(0.1), 
      max_trajectory_points_(1000), publish_rate_(1.0) {
    current_metrics_.timestamp = ros::Time::now();
}

void ExplorationStatistics::initialize(ros::NodeHandle& nh) {
    nh_ = nh;
    
    // 读取参数
    nh_.param("exploration_stats/distance_threshold", min_distance_threshold_, 0.1);
    nh_.param("exploration_stats/max_trajectory_points", max_trajectory_points_, 1000);
    nh_.param("exploration_stats/publish_rate", publish_rate_, 1.0);
    
    double min_area_expansion, efficiency_window, area_resolution;
    nh_.param("exploration_stats/min_area_expansion", min_area_expansion, 1.0);
    nh_.param("exploration_stats/efficiency_window", efficiency_window, 30.0);
    nh_.param("exploration_stats/area_resolution", area_resolution, 0.5);
    
    // 读取里程计话题参数
    std::string odom_topic;
    nh_.param("odometry_topic", odom_topic, std::string("/quad_0/lidar_slam/odom"));
    
    // 订阅里程计
    odom_sub_ = nh_.subscribe(odom_topic, 10, 
                             &ExplorationStatistics::odomCallback, this);
    
    // 发布统计信息
    metrics_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/exploration_metrics", 10);
    
    // 设置发布定时器
    publish_timer_ = nh_.createTimer(ros::Duration(1.0/publish_rate_),
                                    &ExplorationStatistics::publishTimerCallback, this);
    
    start_time_ = ros::Time::now();
    last_update_time_ = start_time_;
    
    ROS_INFO("ExplorationStatistics initialized:");
    ROS_INFO("  - odom topic: '%s'", odom_topic.c_str());
    ROS_INFO("  - publish rate: %.1f Hz", publish_rate_);
    ROS_INFO("  - distance threshold: %.2f m", min_distance_threshold_);
    ROS_INFO("  - area resolution: %.2f m", area_resolution);
    ROS_INFO("  - efficiency window: %.1f s", efficiency_window);
}

void ExplorationStatistics::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    updateOdometry(msg);
}

void ExplorationStatistics::updateOdometry(const nav_msgs::Odometry::ConstPtr& odom) {
    Eigen::Vector3d current_pos(
        odom->pose.pose.position.x,
        odom->pose.pose.position.y,
        odom->pose.pose.position.z
    );
    
    ros::Time current_time = odom->header.stamp;
    
    if (!initialized_) {
        last_position_ = current_pos;
        last_update_time_ = current_time;
        initialized_ = true;
        return;
    }
    
    // 计算累积距离
    calculateDistance(current_pos);
    
    // 更新轨迹点
    trajectory_points_.push_back(current_pos);
    trajectory_times_.push_back(current_time);
    
    // 限制轨迹点数量
    if (trajectory_points_.size() > max_trajectory_points_) {
        trajectory_points_.pop_front();
        trajectory_times_.pop_front();
    }
    
    // 计算速度和时间
    calculateVelocity();
    current_metrics_.exploration_time = (current_time - start_time_).toSec();
    current_metrics_.timestamp = current_time;
    
    // 计算探索效率
    if (current_metrics_.total_distance > 0.001) {
        current_metrics_.exploration_efficiency = 
            current_metrics_.exploration_area / current_metrics_.total_distance;
    }
    
    last_position_ = current_pos;
    last_update_time_ = current_time;
}

void ExplorationStatistics::calculateDistance(const Eigen::Vector3d& new_pos) {
    double distance = (new_pos - last_position_).norm();
    
    // 调试输出 - 定期打印距离信息
    static int debug_counter = 0;
    debug_counter++;
    if (debug_counter % 50 == 0) {  // 每50次更新打印一次
        ROS_INFO("Distance debug: current=%.4f, threshold=%.4f, total=%.3f", 
                 distance, min_distance_threshold_, current_metrics_.total_distance);
    }
    
    // 避免噪声引起的微小移动
    if (distance > min_distance_threshold_) {
        current_metrics_.total_distance += distance;
    }
}

void ExplorationStatistics::calculateVelocity() {
    if (trajectory_points_.size() < 2) {
        current_metrics_.current_velocity = 0.0;
        current_metrics_.average_velocity = 0.0;
        return;
    }
    
    // 计算当前速度（最近两点）
    auto& p1 = trajectory_points_[trajectory_points_.size()-2];
    auto& p2 = trajectory_points_[trajectory_points_.size()-1];
    auto& t1 = trajectory_times_[trajectory_times_.size()-2];
    auto& t2 = trajectory_times_[trajectory_times_.size()-1];
    
    double dt = (t2 - t1).toSec();
    if (dt > 0.001) {
        current_metrics_.current_velocity = (p2 - p1).norm() / dt;
    }
    
    // 计算平均速度
    if (current_metrics_.exploration_time > 0.001) {
        current_metrics_.average_velocity = 
            current_metrics_.total_distance / current_metrics_.exploration_time;
    }
}

void ExplorationStatistics::updateExploredArea(double new_area) {
    current_metrics_.exploration_area = new_area;
    
    // 重新计算效率
    if (current_metrics_.total_distance > 0.001) {
        current_metrics_.exploration_efficiency = 
            current_metrics_.exploration_area / current_metrics_.total_distance;
    }
}

void ExplorationStatistics::updateViewpointCount(int count) {
    current_metrics_.viewpoints_visited = count;
}

void ExplorationStatistics::publishTimerCallback(const ros::TimerEvent& event) {
    publishMetrics();
}

void ExplorationStatistics::publishMetrics() {
    std_msgs::Float64MultiArray msg;
    msg.data = {
        current_metrics_.total_distance,
        current_metrics_.exploration_area,
        current_metrics_.exploration_efficiency,
        current_metrics_.average_velocity,
        current_metrics_.exploration_time,
        static_cast<double>(current_metrics_.viewpoints_visited),
        current_metrics_.current_velocity
    };
    
    // 添加标签信息到layout
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label = "metrics";
    msg.layout.dim[0].size = msg.data.size();
    msg.layout.dim[0].stride = 1;
    
    metrics_pub_.publish(msg);
}

void ExplorationStatistics::reset() {
    current_metrics_ = ExplorationMetrics();
    current_metrics_.timestamp = ros::Time::now();
    trajectory_points_.clear();
    trajectory_times_.clear();
    start_time_ = ros::Time::now();
    last_update_time_ = start_time_;
    initialized_ = false;
    
    ROS_INFO("Exploration statistics reset");
}

std::string ExplorationStatistics::exportMetricsString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "# Exploration Statistics at " << current_metrics_.timestamp << std::endl;
    oss << "total_distance: " << current_metrics_.total_distance << std::endl;
    oss << "exploration_area: " << current_metrics_.exploration_area << std::endl;
    oss << "exploration_efficiency: " << current_metrics_.exploration_efficiency << std::endl;
    oss << "average_velocity: " << current_metrics_.average_velocity << std::endl;
    oss << "current_velocity: " << current_metrics_.current_velocity << std::endl;
    oss << "exploration_time: " << current_metrics_.exploration_time << std::endl;
    oss << "viewpoints_visited: " << current_metrics_.viewpoints_visited << std::endl;
    oss << "trajectory_points_count: " << trajectory_points_.size() << std::endl;
    return oss.str();
}

} // namespace fast_planner
