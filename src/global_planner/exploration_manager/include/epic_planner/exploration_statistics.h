/**
 * @file exploration_statistics.h
 * @brief 探索统计模块 - 统计累积距离、探索面积、效率指标等
 */

#ifndef EXPLORATION_STATISTICS_H
#define EXPLORATION_STATISTICS_H

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Eigen>
#include <vector>
#include <deque>

namespace fast_planner {

struct ExplorationMetrics {
    double total_distance = 0.0;           // 累积移动距离
    double exploration_area = 0.0;         // 探索区域面积
    double exploration_efficiency = 0.0;   // 探索效率 (面积/距离)
    double average_velocity = 0.0;         // 平均速度
    double exploration_time = 0.0;         // 探索总时间
    int viewpoints_visited = 0;            // 访问的视点数量
    double current_velocity = 0.0;         // 当前速度
    ros::Time timestamp;                   // 时间戳
    
    // 格式化输出
    std::string toString() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        oss << "total_distance=" << total_distance 
            << " exploration_area=" << exploration_area
            << " efficiency=" << exploration_efficiency
            << " avg_vel=" << average_velocity
            << " time=" << exploration_time
            << " viewpoints=" << viewpoints_visited;
        return oss.str();
    }
};

class ExplorationStatistics {
public:
    typedef std::shared_ptr<ExplorationStatistics> Ptr;
    
    ExplorationStatistics();
    ~ExplorationStatistics() = default;
    
    // 初始化
    void initialize(ros::NodeHandle& nh);
    
    // 更新位置信息
    void updateOdometry(const nav_msgs::Odometry::ConstPtr& odom);
    
    // 更新探索区域信息
    void updateExploredArea(double new_area);
    
    // 更新访问的视点数量
    void updateViewpointCount(int count);
    
    // 获取当前统计数据
    ExplorationMetrics getCurrentMetrics() const { return current_metrics_; }
    
    // 重置统计
    void reset();
    
    // 导出统计数据到字符串（用于文件保存）
    std::string exportMetricsString() const;

private:
    // ROS相关
    ros::NodeHandle nh_;
    ros::Subscriber odom_sub_;
    ros::Publisher metrics_pub_;
    ros::Timer publish_timer_;
    
    // 统计数据
    ExplorationMetrics current_metrics_;
    std::deque<Eigen::Vector3d> trajectory_points_;  // 轨迹点
    std::deque<ros::Time> trajectory_times_;         // 对应时间戳
    
    Eigen::Vector3d last_position_;
    ros::Time start_time_;
    ros::Time last_update_time_;
    bool initialized_;
    
    // 参数
    double min_distance_threshold_;  // 最小距离阈值（避免噪声）
    int max_trajectory_points_;      // 最大轨迹点数量
    double publish_rate_;            // 发布频率
    
    // 回调函数
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void publishTimerCallback(const ros::TimerEvent& event);
    
    // 内部计算函数
    void calculateDistance(const Eigen::Vector3d& new_pos);
    void calculateVelocity();
    void publishMetrics();
};

} // namespace fast_planner

#endif // EXPLORATION_STATISTICS_H
