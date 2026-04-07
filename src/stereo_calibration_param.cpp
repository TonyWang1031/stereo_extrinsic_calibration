//
// Created by tony on 04/03/26.
//

#include "/home/tony/claude/stereo_extrinsic_calibration/include/stereo_calibration_param.h"

// 构造函数
StereoCalibrationParam::StereoCalibrationParam():EigenInternalMatrixLeft(),
                                                 EigenInternalMatrixRight(),
                                                 DistortionCoeffsLeft(cv::Mat::zeros(1, 5, CV_64F)),
                                                 DistortionCoeffsRight(cv::Mat::zeros(1, 5, CV_64F)),
                                                 EigenExternalMatrixLeft(),
                                                 EigenExternalMatrixRight(),
                                                 MatchedPointsLeft(),
                                                 MatchedPointsRight(),
                                                 PosteriorExternalMatrixRight(),
                                                 ReprojectionError(0.0)
{
}

// 析构函数
StereoCalibrationParam::~StereoCalibrationParam()
{
}

// 获取相机内参
void StereoCalibrationParam::GetInternalMatrix(Eigen::Matrix3d &eigenInternalMatrixLeft,
                                               Eigen::Matrix3d &eigenInternalMatrixRight)
{
    eigenInternalMatrixLeft = EigenInternalMatrixLeft;
    eigenInternalMatrixRight = EigenInternalMatrixRight;
}

// 获取畸变系数
void StereoCalibrationParam::GetDistortionCoeffs(cv::Mat &distortionCoeffsLeft,
                                                  cv::Mat &distortionCoeffsRight)
{
    distortionCoeffsLeft = DistortionCoeffsLeft;
    distortionCoeffsRight = DistortionCoeffsRight;
}

// 获取相机外参
void StereoCalibrationParam::GetExternalMatrix(Eigen::Isometry3d &eigenExternalMatrixLeft,
                                               Eigen::Isometry3d &eigenExternalMatrixRight)
{
    eigenExternalMatrixLeft = EigenExternalMatrixLeft;
    eigenExternalMatrixRight = EigenExternalMatrixRight;
}

// 设置特征匹配点
void StereoCalibrationParam::SetMatchedPoints(std::vector<cv::Point2f> &matchedPointsLeft,
                                              std::vector<cv::Point2f> &matchedPointsRight)
{
    MatchedPointsLeft = matchedPointsLeft;
    MatchedPointsRight = matchedPointsRight;
}

// 获取特征匹配点
void StereoCalibrationParam::GetMatchedPoints(std::vector<cv::Point2f> &matchedPointsLeft,
                                              std::vector<cv::Point2f> &matchedPointsRight)
{
    matchedPointsLeft = MatchedPointsLeft;
    matchedPointsRight = MatchedPointsRight;
}

// 保存标定结果
void StereoCalibrationParam::SaveCalibrationResult(Eigen::Isometry3d &posteriorExternalMatrixRight,
                                                   double &reprojectionError)
{
    PosteriorExternalMatrixRight = posteriorExternalMatrixRight;
    ReprojectionError = reprojectionError;
}

// 获取标定结果
void StereoCalibrationParam::GetCalibrationResult(Eigen::Isometry3d &posteriorExternalMatrixRight,
                                                  double &reprojectionError)
{
    posteriorExternalMatrixRight = PosteriorExternalMatrixRight;
    reprojectionError = ReprojectionError;
}

// 参数文件加载函数
bool StereoCalibrationParam::LoadFromYamlFile(const std::string &yamlFileName,
                                              StereoCalibrationParam *stereoCalibrationParam)
{
    // 判断YAML配置文件是否存在
    if (access(yamlFileName.c_str(), F_OK) == -1)
    {
        return false;
    }

    // 判断YAML配置文件是否可读
    if (access(yamlFileName.c_str(), R_OK) == -1)
    {
        return false;
    }

    // 创建并打开文件存储器
    cv::FileStorage fileStorage;
    if (!fileStorage.open(yamlFileName, cv::FileStorage::READ))
    {
        return false;
    }

    // 读取左相机内参矩阵
    cv::Mat cvInternalMatrixLeft;
    if ((!fileStorage["CvInternalMatrixLeft"].isNone()) && (fileStorage["CvInternalMatrixLeft"].isMap()))
    {
        fileStorage["CvInternalMatrixLeft"] >> cvInternalMatrixLeft;
        cv::cv2eigen(cvInternalMatrixLeft, stereoCalibrationParam -> EigenInternalMatrixLeft);
    }

    // 读取右相机内参矩阵
    cv::Mat cvInternalMatrixRight;
    if ((!fileStorage["CvInternalMatrixRight"].isNone()) && (fileStorage["CvInternalMatrixRight"].isMap()))
    {
        fileStorage["CvInternalMatrixRight"] >> cvInternalMatrixRight;
        cv::cv2eigen(cvInternalMatrixRight, stereoCalibrationParam -> EigenInternalMatrixRight);
    }

    // 读取左相机畸变系数
    if ((!fileStorage["CvDistortionCoeffsLeft"].isNone()) && (fileStorage["CvDistortionCoeffsLeft"].isMap()))
    {
        fileStorage["CvDistortionCoeffsLeft"] >> stereoCalibrationParam -> DistortionCoeffsLeft;
    }

    // 读取右相机畸变系数
    if ((!fileStorage["CvDistortionCoeffsRight"].isNone()) && (fileStorage["CvDistortionCoeffsRight"].isMap()))
    {
        fileStorage["CvDistortionCoeffsRight"] >> stereoCalibrationParam -> DistortionCoeffsRight;
    }

    // 读取左相机外参矩阵
    cv::Mat cvExternalMatrixLeft;
    if ((!fileStorage["CvExternalMatrixLeft"].isNone()) && (fileStorage["CvExternalMatrixLeft"].isMap()))
    {
        fileStorage["CvExternalMatrixLeft"] >> cvExternalMatrixLeft;
        cv::cv2eigen(cvExternalMatrixLeft, stereoCalibrationParam -> EigenExternalMatrixLeft.matrix());
    }

    // 读取右相机外参矩阵
    cv::Mat cvExternalMatrixRight;
    if ((!fileStorage["CvExternalMatrixRight"].isNone()) && (fileStorage["CvExternalMatrixRight"].isMap()))
    {
        fileStorage["CvExternalMatrixRight"] >> cvExternalMatrixRight;
        cv::cv2eigen(cvExternalMatrixRight, stereoCalibrationParam -> EigenExternalMatrixRight.matrix());
    }

    // 关闭文件存储器
    fileStorage.release();

    return true;
}
