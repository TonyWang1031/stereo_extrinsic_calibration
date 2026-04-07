cmake_minimum_required(VERSION 3.0.2)
project(stereo_extrinsic_calibration)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

# 尝试查找catkin，如果找到则构建ROS节点
find_package(catkin QUIET COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge
  image_transport
  message_filters
  tf2_ros
  std_msgs
  geometry_msgs
)

if(catkin_FOUND)
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES calibration
    CATKIN_DEPENDS roscpp sensor_msgs cv_bridge image_transport message_filters tf2_ros std_msgs geometry_msgs
    DEPENDS OpenCV
  )
endif()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

if(catkin_FOUND)
  include_directories(${catkin_INCLUDE_DIRS})
endif()

add_library(calibration
  src/stereo_calibration_param.cpp
  src/stereo_calibration.cpp
)
target_link_libraries(calibration ${OpenCV_LIBS})

# 离线测试可执行文件
add_executable(test_stereo_calibration test/test_stereo_calibration.cpp)
target_link_libraries(test_stereo_calibration calibration ${OpenCV_LIBS})

# 合成测试数据生成器
add_executable(generate_test_data test/generate_test_data.cpp)
target_link_libraries(generate_test_data ${OpenCV_LIBS})

# ROS节点（仅在catkin环境下构建）
if(catkin_FOUND)
  add_executable(stereo_calibration_node src/stereo_calibration_node.cpp)
  target_link_libraries(stereo_calibration_node calibration ${catkin_LIBRARIES} ${OpenCV_LIBS})
endif()
