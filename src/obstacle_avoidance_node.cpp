#include <ros/ros.h>
#include "obstacle_avoidance_test/fusion_avoidance.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "obstacle_avoidance_test");
    ros::NodeHandle nh("~");
    
    // 创建融合避障对象
    obstacle_avoidance::FusionAvoidance avoidance(nh);
    
    // 初始化
    avoidance.initialize();
    
    // 启动算法处理
    avoidance.start();
    
    ROS_INFO("Obstacle avoidance test node started");
    
    // 主循环
    ros::spin();
    
    // 停止处理
    avoidance.stop();
    
    return 0;
}
