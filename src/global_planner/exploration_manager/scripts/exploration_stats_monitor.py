#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
探索统计监控脚本
实时显示探索统计信息
"""

import rospy
from std_msgs.msg import Float64MultiArray
import numpy as np

class ExplorationStatsMonitor:
    def __init__(self):
        rospy.init_node('exploration_stats_monitor', anonymous=True)
        
        # 订阅统计信息
        self.stats_sub = rospy.Subscriber('/exploration_metrics', Float64MultiArray, 
                                         self.stats_callback)
        
        # 统计标签
        self.labels = [
            'Total Distance (m)',
            'Exploration Area (m²)', 
            'Exploration Efficiency (m²/m)',
            'Average Velocity (m/s)',
            'Exploration Time (s)',
            'Viewpoints Visited',
            'Current Velocity (m/s)'
        ]
        
        rospy.loginfo("Exploration Statistics Monitor Started")
        rospy.loginfo("Listening to /exploration_metrics topic...")

    def stats_callback(self, msg):
        """处理统计信息回调"""
        if len(msg.data) >= 7:
            rospy.loginfo("\n" + "="*50)
            rospy.loginfo("EXPLORATION STATISTICS")
            rospy.loginfo("="*50)
            
            for i, (label, value) in enumerate(zip(self.labels, msg.data)):
                if i == 5:  # Viewpoints count - 显示为整数
                    rospy.loginfo("%s: %d" % (label, int(value)))
                else:
                    rospy.loginfo("%s: %.3f" % (label, value))
            
            rospy.loginfo("="*50)
        else:
            rospy.logwarn("Invalid message format, expected 7 values, got %d", len(msg.data))

    def run(self):
        """运行监控器"""
        rospy.spin()

if __name__ == '__main__':
    try:
        monitor = ExplorationStatsMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass
