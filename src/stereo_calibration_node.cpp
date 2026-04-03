#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include "stereo_extrinsic_calibration/feature_matcher.h"
#include "stereo_extrinsic_calibration/stereo_calibrator.h"

using namespace stereo_calibration;

class StereoCalibrationNode {
public:
    StereoCalibrationNode(ros::NodeHandle& nh) {
        // 读取相机内参
        cv::Mat K_left = (cv::Mat_<double>(3, 3) <<
            500, 0, 320,
            0, 500, 240,
            0, 0, 1);
        cv::Mat K_right = K_left.clone();

        // 从ROS参数服务器读取内参
        std::vector<double> k_left_vec, k_right_vec;
        if (nh.getParam("K_left", k_left_vec) && k_left_vec.size() == 9) {
            K_left = cv::Mat(3, 3, CV_64F, k_left_vec.data()).clone();
        }
        if (nh.getParam("K_right", k_right_vec) && k_right_vec.size() == 9) {
            K_right = cv::Mat(3, 3, CV_64F, k_right_vec.data()).clone();
        }

        matcher_ = std::make_unique<FeatureMatcher>(2000);
        calibrator_ = std::make_unique<StereoCalibrator>(K_left, K_right);

        // 订阅双目图像话题
        sub_left_.subscribe(nh, "/camera/left/image_raw", 1);
        sub_right_.subscribe(nh, "/camera/right/image_raw", 1);

        sync_ = std::make_unique<Sync>(SyncPolicy(10), sub_left_, sub_right_);
        sync_->registerCallback(&StereoCalibrationNode::imageCallback, this);

        ROS_INFO("Stereo extrinsic calibration node started");
    }

private:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg_left,
                       const sensor_msgs::ImageConstPtr& msg_right) {
        cv::Mat img_left = cv_bridge::toCvShare(msg_left, "bgr8")->image;
        cv::Mat img_right = cv_bridge::toCvShare(msg_right, "bgr8")->image;

        // 特征匹配
        std::vector<cv::Point2f> pts_left, pts_right;
        int num_matches = matcher_->matchStereo(img_left, img_right, pts_left, pts_right);

        if (num_matches < 20) {
            ROS_WARN("Too few matches: %d", num_matches);
            return;
        }

        // 添加帧进行标定
        if (calibrator_->addFrame(pts_left, pts_right)) {
            cv::Mat R = calibrator_->getRotation();
            cv::Mat t = calibrator_->getTranslation();
            double err = calibrator_->getReprojectionError();

            ROS_INFO("Calibration updated. Reprojection error: %.4f px", err);
            ROS_INFO("R = [%.4f %.4f %.4f; %.4f %.4f %.4f; %.4f %.4f %.4f]",
                     R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
                     R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
                     R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2));
            ROS_INFO("t = [%.4f %.4f %.4f]",
                     t.at<double>(0), t.at<double>(1), t.at<double>(2));
        }
    }

    std::unique_ptr<FeatureMatcher> matcher_;
    std::unique_ptr<StereoCalibrator> calibrator_;

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Sync;

    message_filters::Subscriber<sensor_msgs::Image> sub_left_, sub_right_;
    std::unique_ptr<Sync> sync_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "stereo_calibration_node");
    ros::NodeHandle nh("~");
    StereoCalibrationNode node(nh);
    ros::spin();
    return 0;
}
