#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace stereo_calibration {

/**
 * @brief 特征提取和匹配类
 * 使用ORB特征进行双目图像的特征提取和匹配
 */
class FeatureMatcher {
public:
    FeatureMatcher(int num_features = 2000, float match_ratio = 0.75f);

    /**
     * @brief 提取并匹配双目图像的特征点
     * @param img_left 左图
     * @param img_right 右图
     * @param pts_left 输出：左图匹配点
     * @param pts_right 输出：右图匹配点
     * @return 匹配点对数量
     */
    int matchStereo(const cv::Mat& img_left, const cv::Mat& img_right,
                    std::vector<cv::Point2f>& pts_left,
                    std::vector<cv::Point2f>& pts_right);

private:
    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::BFMatcher> matcher_;
    float match_ratio_;
};

} // namespace stereo_calibration

#endif
