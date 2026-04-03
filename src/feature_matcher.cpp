#include "stereo_extrinsic_calibration/feature_matcher.h"

namespace stereo_calibration {

FeatureMatcher::FeatureMatcher(int num_features, float match_ratio)
    : match_ratio_(match_ratio) {
    // 创建ORB特征检测器
    orb_ = cv::ORB::create(num_features);
    // 创建暴力匹配器（汉明距离）
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);
}

int FeatureMatcher::matchStereo(const cv::Mat& img_left, const cv::Mat& img_right,
                                std::vector<cv::Point2f>& pts_left,
                                std::vector<cv::Point2f>& pts_right) {
    // 1. 提取特征点和描述子
    std::vector<cv::KeyPoint> kpts_left, kpts_right;
    cv::Mat desc_left, desc_right;

    orb_->detectAndCompute(img_left, cv::noArray(), kpts_left, desc_left);
    orb_->detectAndCompute(img_right, cv::noArray(), kpts_right, desc_right);

    if (desc_left.empty() || desc_right.empty()) {
        return 0;
    }

    // 2. 特征匹配
    std::vector<cv::DMatch> matches;
    matcher_->match(desc_left, desc_right, matches);

    // 3. 筛选好的匹配（距离阈值）
    float min_dist = 100.0f;
    for (const auto& m : matches) {
        if (m.distance < min_dist) min_dist = m.distance;
    }

    pts_left.clear();
    pts_right.clear();

    for (const auto& m : matches) {
        if (m.distance < std::max(2.0f * min_dist, 30.0f)) {
            pts_left.push_back(kpts_left[m.queryIdx].pt);
            pts_right.push_back(kpts_right[m.trainIdx].pt);
        }
    }

    return pts_left.size();
}

} // namespace stereo_calibration
