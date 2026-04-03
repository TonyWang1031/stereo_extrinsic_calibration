#include "stereo_extrinsic_calibration/stereo_calibrator.h"
#include <opencv2/calib3d.hpp>

namespace stereo_calibration {

StereoCalibrator::StereoCalibrator(const cv::Mat& K_left, const cv::Mat& K_right)
    : K_left_(K_left.clone()), K_right_(K_right.clone()),
      reproj_error_(0.0), max_history_frames_(10) {
    // 初始化为单位矩阵
    R_ = cv::Mat::eye(3, 3, CV_64F);
    t_ = cv::Mat::zeros(3, 1, CV_64F);
}

bool StereoCalibrator::addFrame(const std::vector<cv::Point2f>& pts_left,
                                const std::vector<cv::Point2f>& pts_right) {
    if (pts_left.size() < 8 || pts_left.size() != pts_right.size()) {
        return false;
    }

    // 如果是第一帧，估计初始位姿
    if (cv::countNonZero(R_ == cv::Mat::eye(3, 3, CV_64F)) == 9) {
        if (!estimateInitialPose(pts_left, pts_right)) {
            return false;
        }
    }

    // 三角化3D点
    std::vector<cv::Point3f> points_3d;
    triangulatePoints(pts_left, pts_right, points_3d);

    // 保存到历史
    points_3d_history_.push_back(points_3d);
    points_left_history_.push_back(pts_left);
    points_right_history_.push_back(pts_right);

    // 限制历史帧数
    if (points_3d_history_.size() > max_history_frames_) {
        points_3d_history_.pop_front();
        points_left_history_.pop_front();
        points_right_history_.pop_front();
    }

    // Bundle Adjustment优化
    if (points_3d_history_.size() >= 3) {
        bundleAdjustment();
    }

    return true;
}

bool StereoCalibrator::estimateInitialPose(const std::vector<cv::Point2f>& pts_left,
                                           const std::vector<cv::Point2f>& pts_right) {
    // 1. 计算本质矩阵（Essential Matrix）
    cv::Mat E = cv::findEssentialMat(pts_left, pts_right, K_left_, cv::RANSAC, 0.999, 1.0);

    if (E.empty()) {
        return false;
    }

    // 2. 从本质矩阵恢复R和t
    cv::Mat R, t, mask;
    int inliers = cv::recoverPose(E, pts_left, pts_right, K_left_, R, t, mask);

    if (inliers < 8) {
        return false;
    }

    R_ = R;
    t_ = t;
    return true;
}

void StereoCalibrator::triangulatePoints(const std::vector<cv::Point2f>& pts_left,
                                         const std::vector<cv::Point2f>& pts_right,
                                         std::vector<cv::Point3f>& points_3d) {
    // 构建投影矩阵
    // P_left = K_left * [I | 0]
    cv::Mat P_left = cv::Mat::zeros(3, 4, CV_64F);
    K_left_.copyTo(P_left(cv::Rect(0, 0, 3, 3)));

    // P_right = K_right * [R | t]
    cv::Mat Rt;
    cv::hconcat(R_, t_, Rt);
    cv::Mat P_right = K_right_ * Rt;

    // 三角化
    cv::Mat points_4d;
    cv::triangulatePoints(P_left, P_right, pts_left, pts_right, points_4d);

    // 齐次坐标转3D坐标
    points_3d.clear();
    for (int i = 0; i < points_4d.cols; ++i) {
        float w = points_4d.at<float>(3, i);
        if (std::abs(w) < 1e-6) continue;

        cv::Point3f pt;
        pt.x = points_4d.at<float>(0, i) / w;
        pt.y = points_4d.at<float>(1, i) / w;
        pt.z = points_4d.at<float>(2, i) / w;

        // 深度检查：只保留在两个相机前方的点
        if (pt.z > 0) {
            cv::Mat pt_right = R_ * cv::Mat(pt) + t_;
            if (pt_right.at<double>(2) > 0) {
                points_3d.push_back(pt);
            }
        }
    }
}

void StereoCalibrator::bundleAdjustment() {
    // 收集所有历史帧的3D-2D对应关系
    std::vector<cv::Point3f> all_pts_3d;
    std::vector<cv::Point2f> all_pts_right;

    for (size_t f = 0; f < points_3d_history_.size(); ++f) {
        for (size_t i = 0; i < points_3d_history_[f].size(); ++i) {
            all_pts_3d.push_back(points_3d_history_[f][i]);
            all_pts_right.push_back(points_right_history_[f][i]);
        }
    }

    if (all_pts_3d.size() < 6) {
        return;
    }

    // PnP + RANSAC 重新估计右相机位姿
    cv::Mat rvec, tvec;
    cv::Rodrigues(R_, rvec);
    tvec = t_.clone();

    cv::Mat inliers_idx;
    cv::solvePnPRansac(all_pts_3d, all_pts_right, K_right_, cv::noArray(),
                       rvec, tvec, true, 300, 2.0, 0.99, inliers_idx);

    if (inliers_idx.rows < 6) {
        return;
    }

    // 收集内点用于精化
    std::vector<cv::Point3f> inlier_pts_3d;
    std::vector<cv::Point2f> inlier_pts_right;
    for (int i = 0; i < inliers_idx.rows; ++i) {
        int idx = inliers_idx.at<int>(i);
        inlier_pts_3d.push_back(all_pts_3d[idx]);
        inlier_pts_right.push_back(all_pts_right[idx]);
    }

    // 用LM方法精化位姿
    cv::solvePnPRefineLM(inlier_pts_3d, inlier_pts_right, K_right_, cv::noArray(),
                         rvec, tvec);

    // 更新外参
    cv::Rodrigues(rvec, R_);
    t_ = tvec;

    // 计算重投影误差
    std::vector<cv::Point2f> reproj_pts;
    cv::projectPoints(inlier_pts_3d, rvec, tvec, K_right_, cv::noArray(), reproj_pts);

    double total_error = 0;
    for (size_t i = 0; i < reproj_pts.size(); ++i) {
        double dx = reproj_pts[i].x - inlier_pts_right[i].x;
        double dy = reproj_pts[i].y - inlier_pts_right[i].y;
        total_error += std::sqrt(dx * dx + dy * dy);
    }
    reproj_error_ = total_error / reproj_pts.size();
}

} // namespace stereo_calibration
