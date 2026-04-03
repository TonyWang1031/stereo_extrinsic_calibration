cmake_minimum_required(VERSION 3.0.2)
project(stereo_extrinsic_calibration)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge
  image_transport
  message_filters
)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs cv_bridge image_transport message_filters
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_library(${PROJECT_NAME}
  src/feature_matcher.cpp
  src/stereo_calibrator.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(stereo_calibration_node src/stereo_calibration_node.cpp)
target_link_libraries(stereo_calibration_node ${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS})
