//
// Created by tony on 04/03/26.
//

#ifndef STEREO_EXTRINSIC_CALIBRATION_STEREO_CALIBRATION_PARAM_H
#define STEREO_EXTRINSIC_CALIBRATION_STEREO_CALIBRATION_PARAM_H

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

///< 双目标定参数类
class StereoCalibrationParam {
public:

    /**
     * @brief 构造函数
     */
    StereoCalibrationParam();

    /**
     * @brief 析构函数
     */
    ~StereoCalibrationParam();

    /**
     * @brief 获取相机内参的函数
     */
    void GetInternalMatrix(Eigen::Matrix3d &eigenInternalMatrixLeft,
                           Eigen::Matrix3d &eigenInternalMatrixRight);

    /**
     * @brief 获取畸变系数的函数
     */
    void GetDistortionCoeffs(cv::Mat &distortionCoeffsLeft,
                             cv::Mat &distortionCoeffsRight);

    /**
     * @brief 获取相机外参的函数
     */
    void GetExternalMatrix(Eigen::Isometry3d &eigenExternalMatrixLeft,
                           Eigen::Isometry3d &eigenExternalMatrixRight);

    /**
     * @brief 设置特征匹配点的函数
     */
    void SetMatchedPoints(std::vector<cv::Point2f> &matchedPointsLeft,
                          std::vector<cv::Point2f> &matchedPointsRight);

    /**
     * @brief 获取特征匹配点的函数
     */
    void GetMatchedPoints(std::vector<cv::Point2f> &matchedPointsLeft,
                          std::vector<cv::Point2f> &matchedPointsRight);

    /**
     * @brief 保存标定结果的函数
     */
    void SaveCalibrationResult(Eigen::Isometry3d &posteriorExternalMatrixRight,
                               double &reprojectionError);

    /**
     * @brief 获取标定结果的函数
     */
    void GetCalibrationResult(Eigen::Isometry3d &posteriorExternalMatrixRight,
                              double &reprojectionError);

    /**
     * @brief 参数文件加载的函数
     */
    static bool LoadFromYamlFile(const std::string &yamlFileName,
                                 StereoCalibrationParam *stereoCalibrationParam);

private:

    // 相机内参
    Eigen::Matrix3d EigenInternalMatrixLeft;                                    ///< Eigen格式的左相机内参矩阵
    Eigen::Matrix3d EigenInternalMatrixRight;                                   ///< Eigen格式的右相机内参矩阵

    // 畸变系数
    cv::Mat DistortionCoeffsLeft;                                              ///< 左相机畸变系数
    cv::Mat DistortionCoeffsRight;                                             ///< 右相机畸变系数

    // 相机外参
    Eigen::Isometry3d EigenExternalMatrixLeft;                                  ///< Eigen格式的左相机外参矩阵
    Eigen::Isometry3d EigenExternalMatrixRight;                                 ///< Eigen格式的右相机外参矩阵

    // 特征匹配点
    std::vector<cv::Point2f> MatchedPointsLeft;                                 ///< 左图像特征匹配点
    std::vector<cv::Point2f> MatchedPointsRight;                                ///< 右图像特征匹配点

    // 标定结果
    Eigen::Isometry3d PosteriorExternalMatrixRight;                             ///< 标定后的右相机外参
    double ReprojectionError;                                                   ///< 重投影误差
};

#endif //STEREO_EXTRINSIC_CALIBRATION_STEREO_CALIBRATION_PARAM_H
