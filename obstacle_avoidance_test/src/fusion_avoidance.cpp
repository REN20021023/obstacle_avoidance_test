#include "obstacle_avoidance_test/fusion_avoidance.h"
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>
#include <sensor_msgs/CameraInfo.h>
#include <visualization_msgs/Marker.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace obstacle_avoidance {

FusionAvoidance::FusionAvoidance(ros::NodeHandle& nh) : 
    nh_(nh), 
    model_loaded_(false), 
    camera_info_received_(false),
    running_(false),
    avoidance_active_(false)
{
    // 加载参数
    nh_.param("algorithm/dbscan_eps", dbscan_eps_, 0.3);
    nh_.param("algorithm/dbscan_min_points", dbscan_min_points_, 10);
    nh_.param("algorithm/voxel_leaf_size", voxel_leaf_size_, 0.05);
    nh_.param("algorithm/max_detection_range", max_detection_range_, 10.0);
    nh_.param("algorithm/min_detection_height", min_detection_height_, -0.5);
    nh_.param("algorithm/max_detection_height", max_detection_height_, 2.0);
    nh_.param("algorithm/vision_detection_thresh", vision_detection_thresh_, 0.5);
    nh_.param("algorithm/fusion_matching_thresh", fusion_matching_thresh_, 1.0);
    nh_.param("algorithm/safe_distance", safe_distance_, 1.5);
    nh_.param("algorithm/emergency_distance", emergency_distance_, 0.8);
    nh_.param("algorithm/max_avoidance_speed", max_avoidance_speed_, 1.0);
    nh_.param("algorithm/obstacle_lifetime", obstacle_lifetime_, 1.0);
    
    // 初始化当前状态
    current_position_ = Eigen::Vector3d::Zero();
    current_velocity_ = Eigen::Vector3d::Zero();
    current_orientation_ = Eigen::Quaterniond::Identity();
}

FusionAvoidance::~FusionAvoidance() {
    stop();
}

void FusionAvoidance::initialize() {
    // 初始化发布器
    lidar_obstacles_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/avoidance_test/lidar/obstacles", 1);
    vision_obstacles_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/avoidance_test/vision/obstacles", 1);
    fused_obstacles_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/avoidance_test/fusion/obstacles", 1);
    avoidance_cmd_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("/avoidance_test/cmd_vel", 1);
    avoidance_vis_pub_ = nh_.advertise<visualization_msgs::Marker>("/avoidance_test/avoidance_vector", 1);
    processed_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/avoidance_test/processed_cloud", 1);
    processed_image_pub_ = nh_.advertise<sensor_msgs::Image>("/avoidance_test/processed_image", 1);
    emergency_stop_pub_ = nh_.advertise<std_msgs::Bool>("/avoidance_test/emergency_stop", 1);
    
    // 加载YOLOv3模型
    std::string model_path, config_path, classes_path;
    nh_.param("vision/model_path", model_path, std::string(""));
    nh_.param("vision/config_path", config_path, std::string(""));
    nh_.param("vision/classes_path", classes_path, std::string(""));
    
    if (!model_path.empty() && !config_path.empty()) {
        try {
            net_ = cv::dnn::readNetFromDarknet(config_path, model_path);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            model_loaded_ = true;
            ROS_INFO("Successfully loaded YOLO model");
            
            // 加载类别名称
            if (!classes_path.empty()) {
                std::ifstream file(classes_path);
                std::string line;
                while (std::getline(file, line)) {
                    if (!line.empty()) {
                        class_names_.push_back(line);
                    }
                }
                ROS_INFO("Loaded %zu class names", class_names_.size());
            } else {
                // 默认COCO类别
                class_names_ = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"};
                ROS_INFO("Using default class names");
            }
        } catch (const cv::Exception& e) {
            ROS_ERROR("Failed to load neural network model: %s", e.what());
        }
    } else {
        ROS_WARN("No neural network model specified, vision detection disabled");
    }
    
    ROS_INFO("FusionAvoidance initialized successfully");
}

void FusionAvoidance::start() {
    if (running_) {
        ROS_WARN("FusionAvoidance is already running");
        return;
    }
    
    // 启动订阅器
    lidar_sub_ = nh_.subscribe("/velodyne_points", 1, &FusionAvoidance::processLidarData, this);
    rgb_sub_ = nh_.subscribe("/camera/rgb/image_raw", 1, &FusionAvoidance::processRGBImage, this);
    depth_sub_ = nh_.subscribe("/camera/depth/image_raw", 1, &FusionAvoidance::processDepthImage, this);
    camera_info_sub_ = nh_.subscribe("/camera/rgb/camera_info", 1, &FusionAvoidance::processCameraInfo, this);
    pose_sub_ = nh_.subscribe("/mavros/local_position/pose", 1, &FusionAvoidance::updatePose, this);
    
    // 启动定时器
    double fusion_rate, avoidance_rate, cleanup_rate;
    nh_.param("algorithm/fusion_rate", fusion_rate, 10.0);
    nh_.param("algorithm/avoidance_rate", avoidance_rate, 20.0);
    nh_.param("algorithm/cleanup_rate", cleanup_rate, 2.0);
    
    fusion_timer_ = nh_.createTimer(ros::Duration(1.0/fusion_rate), 
                                    [this](const ros::TimerEvent&) { fuseObstacles(); });
    avoidance_timer_ = nh_.createTimer(ros::Duration(1.0/avoidance_rate), 
                                       [this](const ros::TimerEvent&) { computeAvoidanceCommand(); });
    cleanup_timer_ = nh_.createTimer(ros::Duration(1.0/cleanup_rate), 
                                    [this](const ros::TimerEvent&) { cleanupObstacles(); });
    
    running_ = true;
    ROS_INFO("FusionAvoidance started");
}

void FusionAvoidance::stop() {
    if (!running_) {
        return;
    }
    
    // 停止所有订阅器和定时器
    lidar_sub_.shutdown();
    rgb_sub_.shutdown();
    depth_sub_.shutdown();
    camera_info_sub_.shutdown();
    pose_sub_.shutdown();
    
    fusion_timer_.stop();
    avoidance_timer_.stop();
    cleanup_timer_.stop();
    
    running_ = false;
    ROS_INFO("FusionAvoidance stopped");
}

void FusionAvoidance::processLidarData(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    // 转换点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    // 滤波并聚类
    filterPointCloud(cloud, filtered_cloud);
    detectLidarObstacles(filtered_cloud);
    
    // 发布处理后的点云
    sensor_msgs::PointCloud2 processed_cloud_msg;
    pcl::toROSMsg(*filtered_cloud, processed_cloud_msg);
    processed_cloud_msg.header = cloud_msg->header;
    processed_cloud_pub_.publish(processed_cloud_msg);
}

void FusionAvoidance::processRGBImage(const sensor_msgs::Image::ConstPtr& img_msg) {
    if (!model_loaded_) {
        return;
    }
    
    try {
        // 转换ROS图像到OpenCV格式
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
        current_image_ = cv_ptr->image;
        
        // 检测目标
        detectVisionObstacles(current_image_);
        
        // 如果有深度信息，计算3D位置
        if (!current_depth_.empty()) {
            project2Dto3D();
        }
        
        // 创建可视化图像
        cv::Mat vis_image = current_image_.clone();
        
        // 绘制检测框
        for (const auto& obstacle : vision_obstacles_) {
            cv::Scalar color(0, 255, 0);  // 绿色
            cv::rectangle(vis_image, obstacle.bbox, color, 2);
            
            std::string label = obstacle.class_name + " " + 
                               std::to_string(static_cast<int>(obstacle.confidence * 100)) + "%";
            
            if (obstacle.has_depth) {
                label += " d=" + std::to_string(static_cast<int>(obstacle.position.z() * 100) / 100.0) + "m";
            }
            
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(vis_image, 
                         cv::Point(obstacle.bbox.x, obstacle.bbox.y - text_size.height - 5),
                         cv::Point(obstacle.bbox.x + text_size.width, obstacle.bbox.y),
                         color, cv::FILLED);
            cv::putText(vis_image, label, 
                       cv::Point(obstacle.bbox.x, obstacle.bbox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
        
        // 发布处理后的图像
        sensor_msgs::ImagePtr processed_img_msg = 
            cv_bridge::CvImage(img_msg->header, "bgr8", vis_image).toImageMsg();
        processed_image_pub_.publish(processed_img_msg);
        
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("CV bridge exception: %s", e.what());
    }
}

void FusionAvoidance::processDepthImage(const sensor_msgs::Image::ConstPtr& depth_msg) {
    try {
        // 转换ROS深度图像到OpenCV格式
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        current_depth_ = cv_ptr->image;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("CV bridge exception: %s", e.what());
    }
}

void FusionAvoidance::processCameraInfo(const sensor_msgs::CameraInfo::ConstPtr& info_msg) {
    if (!camera_info_received_) {
        // 提取相机内参
        camera_matrix_ = cv::Mat(3, 3, CV_64F);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                camera_matrix_.at<double>(i, j) = info_msg->K[i*3 + j];
            }
        }
        
        // 提取畸变系数
        dist_coeffs_ = cv::Mat(1, 5, CV_64F);
        for (int i = 0; i < 5; i++) {
            dist_coeffs_.at<double>(0, i) = info_msg->D[i];
        }
        
        camera_info_received_ = true;
        ROS_INFO("Camera info received");
    }
}

void FusionAvoidance::updatePose(const geometry_msgs::PoseStamped::ConstPtr& pose_msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    // 更新位置和姿态
    current_position_.x() = pose_msg->pose.position.x;
    current_position_.y() = pose_msg->pose.position.y;
    current_position_.z() = pose_msg->pose.position.z;
    
    current_orientation_.w() = pose_msg->pose.orientation.w;
    current_orientation_.x() = pose_msg->pose.orientation.x;
    current_orientation_.y() = pose_msg->pose.orientation.y;
    current_orientation_.z() = pose_msg->pose.orientation.z;
}

void FusionAvoidance::filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in, 
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out) {
    // 体素降采样
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cloud_in);
    voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
    voxel_filter.filter(*cloud_downsampled);
    
    // 距离过滤 - 移除过远的点
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_range_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> range_pass;
    range_pass.setInputCloud(cloud_downsampled);
    range_pass.setFilterFieldName("z");
    range_pass.setFilterLimits(min_detection_height_, max_detection_height_);
    range_pass.filter(*cloud_range_filtered);
    
    // 地面移除 - 使用RANSAC查找并移除地面平面
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.1);
    seg.setInputCloud(cloud_range_filtered);
    seg.segment(*inliers, *coefficients);
    
    if (inliers->indices.size() > 0) {
        extract.setInputCloud(cloud_range_filtered);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_out);
    } else {
        *cloud_out = *cloud_range_filtered;
    }
}

void FusionAvoidance::detectLidarObstacles(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    if (cloud->empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    // 创建KdTree用于聚类
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    
    // 进行欧几里德聚类提取（类似DBSCAN）
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(dbscan_eps_);          // 设置聚类距离阈值
    ec.setMinClusterSize(dbscan_min_points_);     // 设置最小点数阈值
    ec.setMaxClusterSize(25000);                  // 设置最大点数阈值
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);
    
    // 处理每个聚类
    lidar_obstacles_.clear();
    int id = 0;
    
    for (const auto& indices : cluster_indices) {
        // 提取聚类点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : indices.indices) {
            cluster->points.push_back(cloud->points[idx]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;
        
        // 计算质心
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cluster, centroid);
        
        // 计算边界框
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);
        
        // 创建障碍物对象
        LidarObstacle obstacle;
        obstacle.id = id++;
        obstacle.centroid = Eigen::Vector3d(centroid[0], centroid[1], centroid[2]);
        obstacle.dimensions = Eigen::Vector3d(
            max_pt.x - min_pt.x,
            max_pt.y - min_pt.y,
            max_pt.z - min_pt.z
        );
        
        // 计算到传感器的距离
        obstacle.confidence = 0.9;  // 激光雷达置信度较高
        obstacle.timestamp = ros::Time::now();
        
        // 加入障碍物列表
        lidar_obstacles_.push_back(obstacle);
    }
    
    ROS_INFO_THROTTLE(1.0, "Detected %zu obstacles using LiDAR", lidar_obstacles_.size());
}

void FusionAvoidance::detectVisionObstacles(const cv::Mat& image) {
    if (!model_loaded_ || image.empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(data_mutex_);
    vision_obstacles_.clear();
    
    // 准备输入数据
    cv::Mat blob = cv::dnn::blobFromImage(image, 1/255.0, cv::Size(416, 416), 
                                         cv::Scalar(0, 0, 0), true, false);
    
    // 设置网络输入
    net_.setInput(blob);
    
    // 获取输出层名称
    std::vector<std::string> output_layer_names = net_.getUnconnectedOutLayersNames();
    
    // 前向传播
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, output_layer_names);
    
    // 用于存储检测结果的向量
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    // 处理每个输出层
    for (const auto& output : outputs) {
        // 每一行代表一个检测结果
        for (int i = 0; i < output.rows; i++) {
            // 获取各个类别的分数
            cv::Mat scores = output.row(i).colRange(5, output.cols);
            cv::Point class_id_point;
            double confidence;
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &class_id_point);
            
            if (confidence > vision_detection_thresh_) {
                int class_id = class_id_point.x;
                
                // 获取边界框的中心、宽度和高度
                float center_x = output.at<float>(i, 0) * image.cols;
                float center_y = output.at<float>(i, 1) * image.rows;
                float width = output.at<float>(i, 2) * image.cols;
                float height = output.at<float>(i, 3) * image.rows;
                
                // 计算边界框的左上角坐标
                float left = center_x - width / 2;
                float top = center_y - height / 2;
                
                class_ids.push_back(class_id);
                confidences.push_back(static_cast<float>(confidence));
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    // 执行非极大值抑制，消除重叠框
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, vision_detection_thresh_, 0.4, indices);
    
    // 创建视觉障碍物对象
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        
        VisionObstacle obstacle;
        obstacle.bbox = boxes[idx];
        obstacle.confidence = confidences[idx];
        obstacle.id = i;
        obstacle.has_depth = false;
        obstacle.timestamp = ros::Time::now();
        
        // 分配类别名称
        int class_id = class_ids[idx];
        if (class_id < class_names_.size()) {
            obstacle.class_name = class_names_[class_id];
        } else {
            obstacle.class_name = "unknown";
        }
        
        vision_obstacles_.push_back(obstacle);
    }
    
    ROS_INFO_THROTTLE(1.0, "Detected %zu obstacles using Vision", vision_obstacles_.size());
}

void FusionAvoidance::project2Dto3D() {
    if (!camera_info_received_ || current_depth_.empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    for (auto& obstacle : vision_obstacles_) {
        // 获取边界框中心点
        cv::Point center(
            obstacle.bbox.x + obstacle.bbox.width / 2,
            obstacle.bbox.y + obstacle.bbox.height / 2
        );
        
        // 检查点是否在图像边界内
        if (center.x >= 0 && center.x < current_depth_.cols &&
            center.y >= 0 && center.y < current_depth_.rows) {
            
            // 获取中心点的深度值(米)
            float depth = current_depth_.at<float>(center.y, center.x);
            
            // 忽略无效深度值
            if (std::isfinite(depth) && depth > 0) {
                // 去除畸变
                cv::Point2d p_distorted(center.x, center.y);
                std::vector<cv::Point2d> p_distorted_vec = {p_distorted};
                std::vector<cv::Point2d> p_undistorted_vec;
                
                cv::undistortPoints(p_distorted_vec, p_undistorted_vec, camera_matrix_, dist_coeffs_, 
                                   cv::noArray(), camera_matrix_);
                cv::Point2d p_undistorted = p_undistorted_vec[0];
                
                // 转换为3D点（相机坐标系）
                double fx = camera_matrix_.at<double>(0, 0);
                double fy = camera_matrix_.at<double>(1, 1);
                double cx = camera_matrix_.at<double>(0, 2);
                double cy = camera_matrix_.at<double>(1, 2);
                
                double x = (p_undistorted.x - cx) * depth / fx;
                double y = (p_undistorted.y - cy) * depth / fy;
                double z = depth;
                
                obstacle.position = Eigen::Vector3d(x, y, z);
                obstacle.has_depth = true;
            }
        }
    }
}

void FusionAvoidance::fuseObstacles() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    // 创建新的融合障碍物列表
    std::vector<FusedObstacle> new_fused_obstacles;
    ros::Time now = ros::Time::now();
    
    // 首先，将所有激光雷达障碍物添加到融合列表中
    for (const auto& lidar_obs : lidar_obstacles_) {
        FusedObstacle fused_obs;
        fused_obs.position = lidar_obs.centroid;
        fused_obs.dimensions = lidar_obs.dimensions;
        fused_obs.confidence = lidar_obs.confidence;
        fused_obs.source = "lidar";
        fused_obs.id = lidar_obs.id;
        fused_obs.timestamp = lidar_obs.timestamp;
        fused_obs.class_name = "unknown";
        
        new_fused_obstacles.push_back(fused_obs);
    }
    
    // 然后，尝试匹配视觉障碍物与已有的融合障碍物
    for (const auto& vision_obs : vision_obstacles_) {
        if (!vision_obs.has_depth) continue;  // 跳过没有深度信息的视觉障碍物
        
        bool matched = false;
        
        // 尝试匹配
        for (auto& fused_obs : new_fused_obstacles) {
            if ((vision_obs.position - fused_obs.position).norm() < fusion_matching_thresh_) {
                // 匹配成功，更新融合障碍物的类别名称和来源
                fused_obs.class_name = vision_obs.class_name;
                fused_obs.source = "fusion";
                fused_obs.confidence = std::max(fused_obs.confidence, (double)vision_obs.confidence);
                fused_obs.timestamp = now;  // 更新时间戳
                matched = true;
                break;
            }
        }
        
        // 如果没有匹配到任何已有障碍物，创建新的融合障碍物
        if (!matched) {
            FusedObstacle fused_obs;
            fused_obs.position = vision_obs.position;
            
            // 根据类别名称估计尺寸
            if (vision_obs.class_name == "person") {
                fused_obs.dimensions = Eigen::Vector3d(0.5, 0.5, 1.8);
            } else if (vision_obs.class_name == "car" || vision_obs.class_name == "truck") {
                fused_obs.dimensions = Eigen::Vector3d(4.0, 1.8, 1.5);
            } else {
                fused_obs.dimensions = Eigen::Vector3d(0.5, 0.5, 0.5);  // 默认尺寸
            }
            
            fused_obs.confidence = vision_obs.confidence;
            fused_obs.source = "vision";
            fused_obs.id = vision_obs.id + 1000;  // 避免ID冲突
            fused_obs.timestamp = vision_obs.timestamp;
            fused_obs.class_name = vision_obs.class_name;
            
            new_fused_obstacles.push_back(fused_obs);
        }
    }
    
    // 更新融合障碍物列表
    fused_obstacles_ = new_fused_obstacles;
    
    // 发布融合结果
    publishResults();
}

void FusionAvoidance::computeAvoidanceCommand() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    // 重置避障状态
    avoidance_active_ = false;
    avoidance_direction_ = Eigen::Vector3d::Zero();
    double min_time_to_collision = std::numeric_limits<double>::max();
    
    // 紧急停止标志
    bool emergency_stop = false;
    
    // 检查所有融合障碍物的潜在碰撞
    for (const auto& obstacle : fused_obstacles_) {
        double ttc;
        Eigen::Vector3d avoidance_vector;
        
        if (checkCollision(obstacle, ttc, avoidance_vector)) {
            // 如果预测到碰撞
            if (ttc < min_time_to_collision) {
                min_time_to_collision = ttc;
                avoidance_direction_ = avoidance_vector;
                avoidance_active_ = true;
            }
            
            // 检查是否需要紧急停止
            double distance = (obstacle.position - current_position_).norm();
            if (distance < emergency_distance_) {
                emergency_stop = true;
            }
        }
    }
    
    // 发布紧急停止命令
    std_msgs::Bool stop_msg;
    stop_msg.data = emergency_stop;
    emergency_stop_pub_.publish(stop_msg);
    
    // 发布避障命令
    if (avoidance_active_) {
        geometry_msgs::TwistStamped cmd_msg;
        cmd_msg.header.stamp = ros::Time::now();
        cmd_msg.header.frame_id = "world";
        
        // 缩放避障方向到最大速度
        Eigen::Vector3d avoid_vel = avoidance_direction_.normalized() * max_avoidance_speed_;
        
        cmd_msg.twist.linear.x = avoid_vel.x();
        cmd_msg.twist.linear.y = avoid_vel.y();
        cmd_msg.twist.linear.z = avoid_vel.z();
        
        avoidance_cmd_pub_.publish(cmd_msg);
        
        // 可视化避障方向
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();
        marker.ns = "avoidance_vector";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;
        
        marker.pose.position.x = current_position_.x();
        marker.pose.position.y = current_position_.y();
        marker.pose.position.z = current_position_.z();
        
        // 计算四元数以设置箭头方向
        Eigen::Vector3d up(0, 0, 1);
        Eigen::Vector3d direction = avoidance_direction_.normalized();
        Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(up, direction);
        
        marker.pose.orientation.w = q.w();
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();
        
        marker.scale.x = 0.1;
        marker.scale.y = 0.2;
        marker.scale.z = avoidance_direction_.norm();
        
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;
        
        avoidance_vis_pub_.publish(marker);
    }
}

bool FusionAvoidance::checkCollision(const FusedObstacle& obstacle, double& time_to_collision, 
                                   Eigen::Vector3d& avoidance_vector) {
    // 当前速度很小，认为无碰撞风险
    if (current_velocity_.norm() < 0.1) {
        return false;
    }
    
    // 运动方向
    Eigen::Vector3d direction = current_velocity_.normalized();
    
    // 从当前位置到障碍物的向量
    Eigen::Vector3d to_obstacle = obstacle.position - current_position_;
    
    // 将该向量投影到运动方向上
    double projection = to_obstacle.dot(direction);
    
    // 如果障碍物在后方，无碰撞风险
    if (projection < 0) {
        return false;
    }
    
    // 最近接近点
    Eigen::Vector3d closest_point = current_position_ + direction * projection;
    
    // 从最近点到障碍物中心的距离
    double distance = (closest_point - obstacle.position).norm();
    
    // 障碍物半径近似
    double obstacle_radius = obstacle.dimensions.norm() / 2.0;
    
    // 检查距离是否小于安全距离加障碍物半径
    if (distance < (safe_distance_ + obstacle_radius)) {
        // 计算碰撞时间
        time_to_collision = projection / std::max(current_velocity_.norm(), 0.1);
        
        // 计算避障向量（垂直于运动方向）
        Eigen::Vector3d to_closest = closest_point - obstacle.position;
        if (to_closest.norm() < 1e-6) {
            // 如果我们正朝着障碍物中心前进，选择一个随机的垂直方向
            Eigen::Vector3d random_dir = Eigen::Vector3d::Random().normalized();
            avoidance_vector = direction.cross(random_dir).normalized();
        } else {
            // 否则，远离障碍物
            avoidance_vector = to_closest.normalized();
        }
        
        return true;
    }
    
    return false;
}

void FusionAvoidance::cleanupObstacles() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    ros::Time now = ros::Time::now();
    
    // 移除过期的障碍物
    auto is_outdated = [&](const FusedObstacle& obs) {
        return (now - obs.timestamp).toSec() > obstacle_lifetime_;
    };
    
    fused_obstacles_.erase(
        std::remove_if(fused_obstacles_.begin(), fused_obstacles_.end(), is_outdated),
        fused_obstacles_.end());
}

void FusionAvoidance::publishResults() {
    // 发布融合障碍物为标记数组
    visualization_msgs::MarkerArray marker_array;
    
    for (const auto& obstacle : fused_obstacles_) {
        // 创建立方体标记
        visualization_msgs::Marker cube_marker;
        cube_marker.header.frame_id = "world";
        cube_marker.header.stamp = ros::Time::now();
        cube_marker.ns = "fused_obstacles";
        cube_marker.id = obstacle.id;
        cube_marker.type = visualization_msgs::Marker::CUBE;
        cube_marker.action = visualization_msgs::Marker::ADD;
        
        // 设置位置
        cube_marker.pose.position.x = obstacle.position.x();
        cube_marker.pose.position.y = obstacle.position.y();
        cube_marker.pose.position.z = obstacle.position.z();
        
        // 设置方向（单位四元数）
        cube_marker.pose.orientation.w = 1.0;
        
        // 设置尺寸
        cube_marker.scale.x = obstacle.dimensions.x();
        cube_marker.scale.y = obstacle.dimensions.y();
        cube_marker.scale.z = obstacle.dimensions.z();
        
        // 根据来源设置颜色
        if (obstacle.source == "lidar") {
            cube_marker.color.r = 1.0;
            cube_marker.color.g = 0.0;
            cube_marker.color.b = 0.0;
        } else if (obstacle.source == "vision") {
            cube_marker.color.r = 0.0;
            cube_marker.color.g = 1.0;
            cube_marker.color.b = 0.0;
        } else {  // fusion
            cube_marker.color.r = 0.0;
            cube_marker.color.g = 0.0;
            cube_marker.color.b = 1.0;
        }
        cube_marker.color.a = 0.7;
        
        // 创建文本标记
        visualization_msgs::Marker text_marker;
        text_marker.header = cube_marker.header;
        text_marker.ns = "obstacle_labels";
        text_marker.id = obstacle.id;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        
        text_marker.pose = cube_marker.pose;
        text_marker.pose.position.z += obstacle.dimensions.z() / 2.0 + 0.2;
        
        text_marker.text = obstacle.class_name + " (" + obstacle.source + ")";
        
        text_marker.scale.z = 0.3;
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        
        marker_array.markers.push_back(cube_marker);
        marker_array.markers.push_back(text_marker);
    }
    
    if (!marker_array.markers.empty()) {
        fused_obstacles_pub_.publish(marker_array);
    }
}

} // namespace obstacle_avoidance
