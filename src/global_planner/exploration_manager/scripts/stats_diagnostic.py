#!/usr/bin/env python3
"""
探索统计数据实时诊断工具
"""

import rospy
from std_msgs.msg import Float64MultiArray
import time

class StatsDiagnostic:
    def __init__(self):
        rospy.init_node('exploration_stats_diagnostic', anonymous=True)
        
        self.last_metrics = None
        self.start_time = time.time()
        
        # 订阅探索统计话题
        self.stats_sub = rospy.Subscriber('/exploration_metrics', Float64MultiArray, 
                                         self.stats_callback, queue_size=1)
        
        rospy.loginfo("🔍 探索统计诊断工具启动")
        rospy.loginfo("正在监测: /exploration_metrics")
        rospy.loginfo("预期量纲: distance(m), area(m²), efficiency(m), velocity(m/s), time(s)")
        print("\n" + "="*80)
        print("📊 实时统计诊断")
        print("="*80)
        
    def stats_callback(self, msg):
        if len(msg.data) < 6:
            rospy.logwarn("数据不完整，跳过分析")
            return
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # 解析数据
        total_distance = msg.data[0]
        explored_area = msg.data[1]
        efficiency = msg.data[2]  
        avg_velocity = msg.data[3]
        exploration_time = msg.data[4]
        viewpoints = int(msg.data[5])
        
        # 计算变化率
        if self.last_metrics:
            dt = exploration_time - self.last_metrics[4]
            dd = total_distance - self.last_metrics[0]
            velocity_from_distance = dd / dt if dt > 0 else 0
            
            print(f"\n⏱️  时间: {elapsed:.1f}s | 探索时间: {exploration_time:.1f}s")
            print(f"📏 累积距离: {total_distance:.3f} m (Δ: +{dd:.3f} m)")
            print(f"📐 探索面积: {explored_area:.1f} m²")
            print(f"⚡ 效率指标: {efficiency:.3f} m²/m = {efficiency:.3f} m")
            print(f"🏃 平均速度: {avg_velocity:.3f} m/s")
            print(f"💨 瞬时速度: {velocity_from_distance:.3f} m/s (根据距离变化计算)")
            print(f"👁️  视点数量: {viewpoints}")
            
            # 诊断建议
            if total_distance < 5 and exploration_time > 10:
                print("⚠️  距离数值偏小，可能原因:")
                print("   - 距离阈值过大(当前应该是0.01m)")
                print("   - 里程计数据更新频率低")
                print("   - 机器人移动范围有限")
                
            if avg_velocity < 0.1:
                print("⚠️  平均速度偏小，检查:")
                print("   - 总距离是否统计完整")
                print("   - 探索时间是否包含了静止时间")
                
        else:
            print(f"🚀 首次数据: 距离={total_distance:.3f}m, 速度={avg_velocity:.3f}m/s")
            
        self.last_metrics = msg.data[:]
        print("-" * 80)
    
    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\n🔚 诊断结束")

if __name__ == '__main__':
    try:
        diagnostic = StatsDiagnostic()
        diagnostic.run()
    except rospy.ROSInterruptException:
        pass
