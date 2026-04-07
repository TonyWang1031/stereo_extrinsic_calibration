//
// Created by tony on 04/03/26.
//

#include "/home/tony/claude/stereo_extrinsic_calibration/include/stereo_calibration.h"
#include "/home/tony/claude/stereo_extrinsic_calibration/include/stereo_calibration_param.h"
#include <iostream>

int main(int argc, char **argv)
{
    // 加载参数
    StereoCalibrationParam stereoCalibrationParam;
    if (!StereoCalibrationParam::LoadFromYamlFile("/home/tony/claude/stereo_extrinsic_calibration/config/camera_params.yaml",
                                                   &stereoCalibrationParam))
    {
        std::cout << "参数文件加载失败" << std::endl;
        return -1;
    }

    // 读取测试图像
    cv::Mat imageLeft = cv::imread("/home/tony/claude/stereo_extrinsic_calibration/data/left.png", cv::IMREAD_GRAYSCALE);
    cv::Mat imageRight = cv::imread("/home/tony/claude/stereo_extrinsic_calibration/data/right.png", cv::IMREAD_GRAYSCALE);

    if (imageLeft.empty() || imageRight.empty())
    {
        std::cout << "图像加载失败" << std::endl;
        return -1;
    }

    // 创建标定对象
    StereoCalibration stereoCalibration;

    // 特征匹配
    std::vector<cv::Point2f> matchedPointsLeft, matchedPointsRight;
    stereoCalibration.FeatureMatching(imageLeft, imageRight, matchedPointsLeft, matchedPointsRight);

    std::cout << "匹配点数量: " << matchedPointsLeft.size() << std::endl;

    if (matchedPointsLeft.size() < 20)
    {
        std::cout << "匹配点数量不足" << std::endl;
        return -1;
    }

    // 设置匹配点
    stereoCalibrationParam.SetMatchedPoints(matchedPointsLeft, matchedPointsRight);

    // 执行单帧标定（包含RANSAC + 5-DOF优化 + 尺度恢复）
    stereoCalibration.SetPriorParameter(stereoCalibrationParam);
    stereoCalibration.Calibrate();

    // 获取单帧标定结果
    StereoCalibrationParam posteriorParam;
    stereoCalibration.GetPosteriorParameter(posteriorParam);

    Eigen::Isometry3d posteriorExternalMatrixRight;
    double epipolarError;
    posteriorParam.GetCalibrationResult(posteriorExternalMatrixRight, epipolarError);

    std::cout << "\n=== 单帧标定结果 ===" << std::endl;
    std::cout << "极线误差(RMS): " << epipolarError << std::endl;
    std::cout << "旋转矩阵:\n" << posteriorExternalMatrixRight.rotation() << std::endl;
    std::cout << "平移向量: " << posteriorExternalMatrixRight.translation().transpose() << std::endl;

    // 多帧融合标定演示
    std::cout << "\n=== 多帧融合标定 ===" << std::endl;
    stereoCalibration.CalibrateMultiFrame();

    stereoCalibration.GetPosteriorParameter(posteriorParam);
    posteriorParam.GetCalibrationResult(posteriorExternalMatrixRight, epipolarError);

    std::cout << "融合后极线误差(RMS): " << epipolarError << std::endl;
    std::cout << "融合后旋转矩阵:\n" << posteriorExternalMatrixRight.rotation() << std::endl;
    std::cout << "融合后平移向量: " << posteriorExternalMatrixRight.translation().transpose() << std::endl;

    return 0;
}