#ifndef FUSION_AVOIDANCE_H
#define FUSION_AVOIDANCE_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Bool.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <Eigen/Dense>
#include <mutex>
#include <vector>
#include <memory>

namespace obstacle_avoidance {

// 定义点云障碍物结构
struct LidarObstacle {
    Eigen::Vector3d centroid;
    Eigen::Vector3d dimensions;
    int id;
    double confidence;
    ros::Time timestamp;
};

// 定义视觉障碍物结构
struct VisionObstacle {
    cv::Rect bbox;
    std::string class_name;
    float confidence;
    int id;
    Eigen::Vector3d position;
    bool has_depth;
    ros::Time timestamp;
};

// 定义融合后的障碍物结构
struct FusedObstacle {
    Eigen::Vector3d position;
    Eigen::Vector3d dimensions;
    double confidence;
    std::string source;  // "lidar", "vision", "fusion"
    std::string class_name;
    int id;
    ros::Time timestamp;
};

class FusionAvoidance {
public:
    FusionAvoidance(ros::NodeHandle& nh);
    ~FusionAvoidance();

    // 初始化函数
    void initialize();
    
    // 启动/停止处理
    void start();
    void stop();
    
private:
    // 主处理函数
    void processLidarData(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg);
    void processRGBImage(const sensor_msgs::Image::ConstPtr& img_msg);
    void processDepthImage(const sensor_msgs::Image::ConstPtr& depth_msg);
    void processCameraInfo(const sensor_msgs::CameraInfo::ConstPtr& info_msg);
    void updatePose(const geometry_msgs::PoseStamped::ConstPtr& pose_msg);
    
    // 核心算法
    void detectLidarObstacles(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void detectVisionObstacles(const cv::Mat& image);
    void project2Dto3D();
    void fuseObstacles();
    void computeAvoidanceCommand();
    
    // 点云处理
    void filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in, 
                         pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out);
    void performDBSCANClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in);
    
    // 辅助函数
    bool checkCollision(const FusedObstacle& obstacle, double& time_to_collision, 
                       Eigen::Vector3d& avoidance_vector);
    void cleanupObstacles();
    
    // 可视化函数
    void publishResults();
    
    // ROS接口
    ros::NodeHandle nh_;
    
    // 订阅器
    ros::Subscriber lidar_sub_;
    ros::Subscriber rgb_sub_;
    ros::Subscriber depth_sub_;
    ros::Subscriber camera_info_sub_;
    ros::Subscriber pose_sub_;
    
    // 发布器
    ros::Publisher lidar_obstacles_pub_;
    ros::Publisher vision_obstacles_pub_;
    ros::Publisher fused_obstacles_pub_;
    ros::Publisher avoidance_cmd_pub_;
    ros::Publisher avoidance_vis_pub_;
    ros::Publisher processed_cloud_pub_;
    ros::Publisher processed_image_pub_;
    ros::Publisher emergency_stop_pub_;
    
    // 定时器
    ros::Timer fusion_timer_;
    ros::Timer avoidance_timer_;
    ros::Timer cleanup_timer_;
    
    // 算法参数
    double dbscan_eps_;              // DBSCAN聚类距离阈值
    int dbscan_min_points_;          // DBSCAN最小点数
    double voxel_leaf_size_;         // 体素滤波尺寸
    double max_detection_range_;     // 最大检测范围
    double min_detection_height_;    // 最小检测高度
    double max_detection_height_;    // 最大检测高度
    double vision_detection_thresh_; // 视觉检测阈值
    double fusion_matching_thresh_;  // 融合匹配阈值
    double safe_distance_;           // 安全距离
    double emergency_distance_;      // 紧急停止距离
    double max_avoidance_speed_;     // 最大避障速度
    double obstacle_lifetime_;       // 障碍物寿命
    
    // 算法内部状态
    std::vector<LidarObstacle> lidar_obstacles_;
    std::vector<VisionObstacle> vision_obstacles_;
    std::vector<FusedObstacle> fused_obstacles_;
    
    // 当前状态
    Eigen::Vector3d current_position_;
    Eigen::Vector3d current_velocity_;
    Eigen::Quaterniond current_orientation_;
    bool avoidance_active_;
    Eigen::Vector3d avoidance_direction_;
    
    // OpenCV/PCL对象
    cv::Mat current_image_;
    cv::Mat current_depth_;
    cv::dnn::Net net_;
    std::vector<std::string> class_names_;
    bool model_loaded_;
    
    // 相机参数
    bool camera_info_received_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    
    // 线程安全
    std::mutex data_mutex_;
    
    // 处理运行状态
    bool running_;
};

} // namespace obstacle_avoidance

#endif // FUSION_AVOIDANCE_H
