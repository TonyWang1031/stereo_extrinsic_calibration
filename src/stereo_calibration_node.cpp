//
// ROS节点: 在线双目外参标定
// 订阅左右图像，执行特征匹配和标定，发布外参TF和极线误差
//

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/TransformStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include "stereo_calibration.h"
#include "stereo_calibration_param.h"

class StereoCalibrationNode
{
public:
    StereoCalibrationNode(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    {
        // 加载配置文件路径
        std::string configFile;
        pnh.param<std::string>("config_file", configFile,
                               ros::package::getPath("stereo_extrinsic_calibration") + "/config/camera_params.yaml");

        if (!StereoCalibrationParam::LoadFromYamlFile(configFile, &Param))
        {
            ROS_ERROR("Failed to load config file: %s", configFile.c_str());
            return;
        }
        ROS_INFO("Loaded config from: %s", configFile.c_str());

        // 发布极线误差
        ErrorPub = nh.advertise<std_msgs::Float64>("stereo_calibration/epipolar_error", 10);

        // 订阅左右图像（近似时间同步）
        LeftSub.subscribe(nh, "/camera/left/image_raw", 1);
        RightSub.subscribe(nh, "/camera/right/image_raw", 1);

        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
        Sync.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), LeftSub, RightSub));
        Sync->registerCallback(boost::bind(&StereoCalibrationNode::ImageCallback, this, _1, _2));

        ROS_INFO("Stereo calibration node started.");
    }

private:
    void ImageCallback(const sensor_msgs::ImageConstPtr &leftMsg,
                        const sensor_msgs::ImageConstPtr &rightMsg)
    {
        // 转换图像
        cv::Mat imageLeft, imageRight;
        try
        {
            imageLeft = cv_bridge::toCvShare(leftMsg, "mono8")->image;
            imageRight = cv_bridge::toCvShare(rightMsg, "mono8")->image;
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // 特征匹配
        std::vector<cv::Point2f> matchedPointsLeft, matchedPointsRight;
        Calibrator.FeatureMatching(imageLeft, imageRight, matchedPointsLeft, matchedPointsRight);

        if (matchedPointsLeft.size() < 20)
        {
            ROS_WARN("Not enough matched points: %zu", matchedPointsLeft.size());
            return;
        }

        // 设置匹配点并执行标定
        Param.SetMatchedPoints(matchedPointsLeft, matchedPointsRight);
        Calibrator.SetPriorParameter(Param);
        Calibrator.Calibrate();

        // 多帧融合
        Calibrator.CalibrateMultiFrame();

        // 获取结果
        StereoCalibrationParam posteriorParam;
        Calibrator.GetPosteriorParameter(posteriorParam);

        Eigen::Isometry3d extrinsic;
        double epipolarError;
        posteriorParam.GetCalibrationResult(extrinsic, epipolarError);

        // 发布极线误差
        std_msgs::Float64 errorMsg;
        errorMsg.data = epipolarError;
        ErrorPub.publish(errorMsg);

        // 发布TF (左相机到右相机)
        geometry_msgs::TransformStamped tfMsg;
        tfMsg.header.stamp = leftMsg->header.stamp;
        tfMsg.header.frame_id = "camera_left";
        tfMsg.child_frame_id = "camera_right";

        Eigen::Matrix3d R = extrinsic.rotation();
        Eigen::Vector3d t = extrinsic.translation();

        tfMsg.transform.translation.x = t.x();
        tfMsg.transform.translation.y = t.y();
        tfMsg.transform.translation.z = t.z();

        // 旋转矩阵转四元数
        Eigen::Quaterniond q(R);
        tfMsg.transform.rotation.x = q.x();
        tfMsg.transform.rotation.y = q.y();
        tfMsg.transform.rotation.z = q.z();
        tfMsg.transform.rotation.w = q.w();

        TfBroadcaster.sendTransform(tfMsg);

        ROS_INFO("Epipolar error: %.6f, t: [%.4f, %.4f, %.4f]",
                 epipolarError, t.x(), t.y(), t.z());
    }

    StereoCalibration Calibrator;
    StereoCalibrationParam Param;

    ros::Publisher ErrorPub;
    tf2_ros::TransformBroadcaster TfBroadcaster;

    message_filters::Subscriber<sensor_msgs::Image> LeftSub;
    message_filters::Subscriber<sensor_msgs::Image> RightSub;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> Sync;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "stereo_calibration_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    StereoCalibrationNode node(nh, pnh);

    ros::spin();
    return 0;
}
