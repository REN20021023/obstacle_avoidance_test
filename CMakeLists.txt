cmake_minimum_required(VERSION 3.0.2)
project(obstacle_avoidance_test)

# 编译选项
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -DBOOST_BIND_GLOBAL_PLACEHOLDERS")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

# 查找依赖包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  image_transport
)

find_package(Boost REQUIRED COMPONENTS system thread)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Catkin定义
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES obstacle_avoidance_test
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs geometry_msgs visualization_msgs cv_bridge tf2 tf2_ros
  DEPENDS PCL OpenCV EIGEN3
)

# 包含目录
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# 构建库文件
add_library(fusion_avoidance
  src/fusion_avoidance.cpp
)

target_link_libraries(fusion_avoidance
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
)

# 构建可执行文件
add_executable(obstacle_avoidance_node
  src/obstacle_avoidance_node.cpp
)

target_link_libraries(obstacle_avoidance_node
  fusion_avoidance
  ${catkin_LIBRARIES}
)

# 确保Python脚本可执行
catkin_install_python(PROGRAMS
  scripts/velocity_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装目标
install(TARGETS fusion_avoidance obstacle_avoidance_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装头文件
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# 安装启动文件和配置文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds
)

install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
)
