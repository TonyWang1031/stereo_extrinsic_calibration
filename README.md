# 高精度在线无标记的双目外参标定系统

本工程实现了论文 "High-Precision Online Markerless Stereo Extrinsic Calibration" 中提出的双目在线标定算法。

## 算法原理

### 1. 特征提取与匹配
- 使用 ORB 特征检测器提取双目图像的特征点
- 通过汉明距离进行特征匹配，并使用距离阈值筛选优质匹配

### 2. 初始位姿估计
- 通过本质矩阵（Essential Matrix）估计双目相机的相对位姿
- 使用 RANSAC 提高鲁棒性，过滤外点
- 从本质矩阵分解得到旋转矩阵 R 和平移向量 t

### 3. 三角化重建
- 根据当前外参和双目匹配点进行三角化
- 重建场景的 3D 点云
- 过滤深度为负的无效点

### 4. Bundle Adjustment 优化
- 收集多帧历史数据的 3D-2D 对应关系
- 使用 PnP + RANSAC 重新估计右相机位姿
- 通过 Levenberg-Marquardt 方法精化外参
- 计算重投影误差评估标定精度

### 5. 在线持续优化
- 维护固定长度的历史帧队列（默认10帧）
- 每添加新帧都进行 Bundle Adjustment
- 实现在线实时标定

## 依赖安装

```bash
# ROS (Noetic/Melodic)
sudo apt-get install ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-image-transport

# OpenCV 4
sudo apt-get install libopencv-dev

# Eigen3
sudo apt-get install libeigen3-dev
```

## 编译

```bash
cd ~/catkin_ws/src
git clone <this_repo>
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## 配置

修改 `config/camera_params.yaml`，填入你的相机内参矩阵。

## 运行

```bash
roslaunch stereo_extrinsic_calibration stereo_calibration.launch
```

修改 launch 文件中的图像话题名称以匹配你的相机。

## 输出

节点会实时输出：
- 匹配点数量
- 标定的旋转矩阵 R
- 标定的平移向量 t
- 重投影误差（像素）

重投影误差越小，标定精度越高（通常 < 1.0 像素为优秀）。

## 代码结构

```
include/stereo_extrinsic_calibration/
├── feature_matcher.h      # 特征提取与匹配
└── stereo_calibrator.h    # 双目标定核心算法

src/
├── feature_matcher.cpp
├── stereo_calibrator.cpp
└── stereo_calibration_node.cpp  # ROS节点
```

## 论文参考

详见 `document/High-Precision Online Markerless Stereo Extrinsic Calibration.pdf`
