#ifndef STEREO_CALIBRATOR_H
#define STEREO_CALIBRATOR_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <deque>

namespace stereo_calibration {

/**
 * @brief 双目外参标定类
 * 实现论文中的在线标定算法
 */
class StereoCalibrator {
public:
    StereoCalibrator(const cv::Mat& K_left, const cv::Mat& K_right);

    /**
     * @brief 添加一帧图像对进行标定
     * @param pts_left 左图特征点
     * @param pts_right 右图特征点
     * @return 是否标定成功
     */
    bool addFrame(const std::vector<cv::Point2f>& pts_left,
                  const std::vector<cv::Point2f>& pts_right);

    /**
     * @brief 获取当前标定的旋转矩阵
     */
    cv::Mat getRotation() const { return R_.clone(); }

    /**
     * @brief 获取当前标定的平移向量
     */
    cv::Mat getTranslation() const { return t_.clone(); }

    /**
     * @brief 获取标定精度（重投影误差）
     */
    double getReprojectionError() const { return reproj_error_; }

private:
    // 相机内参
    cv::Mat K_left_, K_right_;

    // 外参：右相机相对左相机的位姿
    cv::Mat R_, t_;

    // 历史3D点和观测
    std::deque<std::vector<cv::Point3f>> points_3d_history_;
    std::deque<std::vector<cv::Point2f>> points_left_history_;
    std::deque<std::vector<cv::Point2f>> points_right_history_;

    double reproj_error_;
    int max_history_frames_;

    /**
     * @brief 通过本质矩阵估计初始位姿
     */
    bool estimateInitialPose(const std::vector<cv::Point2f>& pts_left,
                             const std::vector<cv::Point2f>& pts_right);

    /**
     * @brief 三角化重建3D点
     */
    void triangulatePoints(const std::vector<cv::Point2f>& pts_left,
                          const std::vector<cv::Point2f>& pts_right,
                          std::vector<cv::Point3f>& points_3d);

    /**
     * @brief Bundle Adjustment优化外参
     */
    void bundleAdjustment();
};

} // namespace stereo_calibration

#endif
