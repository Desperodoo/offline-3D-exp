#!/usr/bin/env python3
"""
探索统计监控节点
监听探索统计信息并显示实时数据
"""

import rospy
from std_msgs.msg import Float64MultiArray
import sys

class StatsMonitor:
    def __init__(self):
        rospy.init_node('exploration_stats_monitor', anonymous=True)
        
        # 订阅探索统计话题
        self.stats_sub = rospy.Subscriber('/exploration_metrics', Float64MultiArray, 
                                         self.stats_callback, queue_size=1)
        
        rospy.loginfo("探索统计监控节点启动，监听话题: /exploration_metrics")
        rospy.loginfo("等待探索统计数据...")
        
    def stats_callback(self, msg):
        """处理探索统计数据"""
        try:
            if len(msg.data) < 6:
                rospy.logwarn("收到的统计数据不完整")
                return
                
            # 解析数据 (与ExplorationStatistics::publishMetrics对应)
            total_distance = msg.data[0]
            total_time = msg.data[1] 
            explored_area = msg.data[2]
            efficiency = msg.data[3]
            avg_speed = msg.data[4]
            viewpoints_count = msg.data[5]
            
            # 格式化输出
            print("\n" + "="*60)
            print("📊 探索统计实时数据")
            print("="*60)
            print(f"🚀 累积飞行距离: {total_distance:.2f} m")
            print(f"⏱️  总探索时间:    {total_time:.1f} s")
            print(f"📍 已探索区域:    {explored_area:.2f} m²")
            print(f"⚡ 探索效率:      {efficiency:.3f} m²/m")
            print(f"🏃 平均速度:      {avg_speed:.2f} m/s")
            print(f"👁️  视点数量:      {int(viewpoints_count)}")
            print("="*60)
            
        except Exception as e:
            rospy.logerr(f"处理统计数据时出错: {e}")
    
    def run(self):
        """运行监控节点"""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("探索统计监控节点关闭")

if __name__ == '__main__':
    try:
        monitor = StatsMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass
