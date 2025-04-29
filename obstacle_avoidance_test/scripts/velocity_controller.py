#!/usr/bin/env python

import rospy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool

class VelocityController:
    def __init__(self):
        rospy.init_node('velocity_controller', anonymous=True)
        
        # 初始化参数
        self.emergency_stop = False
        self.last_avoidance_cmd = None
        self.avoidance_timeout = rospy.Duration(0.5)  # 0.5秒超时
        self.last_avoidance_time = rospy.Time.now()
        
        # 订阅话题
        rospy.Subscriber('/cmd_vel_input', TwistStamped, self.input_callback)
        rospy.Subscriber('/cmd_vel_avoidance', TwistStamped, self.avoidance_callback)
        rospy.Subscriber('/emergency_stop', Bool, self.emergency_callback)
        
        # 发布话题
        self.cmd_pub = rospy.Publisher('/cmd_vel', TwistStamped, queue_size=10)
        
        # 定时器
        rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        
        rospy.loginfo("Velocity controller started")
    
    def input_callback(self, msg):
        """处理用户输入的速度命令"""
        # 如果没有紧急停止且当前没有避障命令，则直接转发
        if not self.emergency_stop and (self.last_avoidance_cmd is None or 
                                      (rospy.Time.now() - self.last_avoidance_time) > self.avoidance_timeout):
            self.cmd_pub.publish(msg)
    
    def avoidance_callback(self, msg):
        """处理避障算法发出的速度命令"""
        # 记录避障命令和时间
        self.last_avoidance_cmd = msg
        self.last_avoidance_time = rospy.Time.now()
        
        # 如果没有紧急停止，则发布避障命令
        if not self.emergency_stop:
            self.cmd_pub.publish(msg)
    
    def emergency_callback(self, msg):
        """处理紧急停止命令"""
        self.emergency_stop = msg.data
        
        # 如果接收到紧急停止命令，立即发送零速度命令
        if self.emergency_stop:
            zero_cmd = TwistStamped()
            zero_cmd.header.stamp = rospy.Time.now()
            zero_cmd.header.frame_id = "world"
            self.cmd_pub.publish(zero_cmd)
            rospy.logwarn("Emergency stop activated!")
    
    def timer_callback(self, event):
        """定时检查并保证机器人运动安全"""
        # 如果有紧急停止，确保机器人不移动
        if self.emergency_stop:
            zero_cmd = TwistStamped()
            zero_cmd.header.stamp = rospy.Time.now()
            zero_cmd.header.frame_id = "world"
            self.cmd_pub.publish(zero_cmd)
        
        # 避障命令优先于普通控制命令
        elif self.last_avoidance_cmd is not None and (rospy.Time.now() - self.last_avoidance_time) <= self.avoidance_timeout:
            self.cmd_pub.publish(self.last_avoidance_cmd)

if __name__ == '__main__':
    try:
        controller = VelocityController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
