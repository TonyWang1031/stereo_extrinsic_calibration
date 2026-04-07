//
// Created by tony on 04/03/26.
//

#ifndef STEREO_EXTRINSIC_CALIBRATION_STEREO_CALIBRATION_H
#define STEREO_EXTRINSIC_CALIBRATION_STEREO_CALIBRATION_H

#include <iostream>
#include <vector>
#include <random>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "stereo_calibration_param.h"

///< 双目外参标定类
class StereoCalibration {
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief 构造函数
     */
    StereoCalibration();

    /**
     * @brief 析构函数
     */
    ~StereoCalibration();

    /**
     * @brief 特征提取与匹配
     */
    void FeatureMatching(const cv::Mat &imageLeft,
                         const cv::Mat &imageRight,
                         std::vector<cv::Point2f> &matchedPointsLeft,
                         std::vector<cv::Point2f> &matchedPointsRight);

    /**
     * @brief 本质矩阵估计初始位姿
     * @detail 先用各自内参归一化匹配点，再计算本质矩阵并分解得到R和t
     *         支持左右相机内参不同的情况
     */
    bool EstimateInitialPose(const std::vector<cv::Point2f> &matchedPointsLeft,
                             const std::vector<cv::Point2f> &matchedPointsRight,
                             const Eigen::Matrix3d &internalMatrixLeft,
                             const Eigen::Matrix3d &internalMatrixRight,
                             const cv::Mat &distortionCoeffsLeft,
                             const cv::Mat &distortionCoeffsRight,
                             Eigen::Matrix3d &rotationMatrix,
                             Eigen::Vector3d &translationVector);

    /**
     * @brief 三角化重建3D点
     */
    void TriangulatePoints(const std::vector<cv::Point2f> &matchedPointsLeft,
                           const std::vector<cv::Point2f> &matchedPointsRight,
                           const Eigen::Matrix3d &internalMatrixLeft,
                           const Eigen::Matrix3d &internalMatrixRight,
                           const Eigen::Matrix3d &rotationMatrix,
                           const Eigen::Vector3d &translationVector,
                           std::vector<Eigen::Vector3d> &points3D);

    /**
     * @brief 5自由度非线性优化（论文Algorithm 1）
     * @detail 基于极线约束的Gauss-Newton迭代，优化变量 ξ = [δθ; δt] ∈ ℝ⁵
     *         旋转用李代数参数化(3-DOF)，平移用切空间参数化(2-DOF)
     */
    void NonlinearOptimization(const std::vector<Eigen::Vector3d> &normalizedPointsLeft,
                               const std::vector<Eigen::Vector3d> &normalizedPointsRight,
                               Eigen::Matrix3d &rotationMatrix,
                               Eigen::Vector3d &translationVector,
                               double &epipolarError);

    /**
     * @brief 设置先验参数函数
     */
    void SetPriorParameter(StereoCalibrationParam &priorParameter);

    /**
     * @brief 得到后验参数函数
     */
    void GetPosteriorParameter(StereoCalibrationParam &posteriorParameter);

    /**
     * @brief RANSAC外点剔除 + 5-DOF优化（论文Section V-B）
     * @detail 随机采样最小集(5对点) → 5-DOF优化 → 计算极线误差 → 统计内点
     *         → 选最优模型 → 用全部内点重新优化
     * @param[out] inlierIndices 内点索引
     */
    void RansacOptimization(const std::vector<Eigen::Vector3d> &normalizedPointsLeft,
                            const std::vector<Eigen::Vector3d> &normalizedPointsRight,
                            Eigen::Matrix3d &rotationMatrix,
                            Eigen::Vector3d &translationVector,
                            double &epipolarError,
                            std::vector<int> &inlierIndices);

    /**
     * @brief 多帧融合标定（论文Section V-C）
     * @detail 累积多帧归一化匹配点，联合优化 C = Σₖ Σᵢ ||rᵢᵏ||²
     */
    void AccumulateFrame(const std::vector<Eigen::Vector3d> &normalizedPointsLeft,
                         const std::vector<Eigen::Vector3d> &normalizedPointsRight);

    /**
     * @brief 执行多帧融合标定
     */
    void CalibrateMultiFrame();

    /**
     * @brief 执行标定
     */
    void Calibrate();

private:

    StereoCalibrationParam PriorParameter;                                      ///< 先验参数
    StereoCalibrationParam PosteriorParameter;                                  ///< 后验参数

    cv::Ptr<cv::ORB> OrbDetector;                                               ///< ORB特征检测器
    cv::Ptr<cv::BFMatcher> BfMatcher;                                           ///< 暴力匹配器

    static const int MaxWindowSize = 10;                                         ///< 滑动窗口最大帧数M
    std::vector<std::vector<Eigen::Vector3d>> FrameBufferLeft;                   ///< 左归一化点帧缓冲区
    std::vector<std::vector<Eigen::Vector3d>> FrameBufferRight;                  ///< 右归一化点帧缓冲区
    Eigen::Matrix3d CurrentRotation;                                             ///< 当前旋转估计
    Eigen::Vector3d CurrentTranslation;                                          ///< 当前平移估计
    bool IsInitialized;                                                          ///< 是否已初始化

    /**
     * @brief 构造反对称矩阵 [v]×
     * @detail 将向量 v = [v₁, v₂, v₃]ᵀ 映射为 3×3 反对称矩阵
     */
    Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d &v);

    /**
     * @brief 指数映射 exp([δθ]×)（Rodrigues公式）
     * @detail 将李代数 so(3) 映射到旋转群 SO(3)
     */
    Eigen::Matrix3d ExpMap(const Eigen::Vector3d &deltaTheta);

    /**
     * @brief 构造切空间正交基（论文Algorithm 2）
     * @detail 在单位球面 S² 上点 t 处构造切平面的两个正交基 b₁, b₂
     *         满足 b₁⊥t, b₂⊥t, b₁⊥b₂, ||b₁||=||b₂||=1
     */
    void FindingBases(const Eigen::Vector3d &translationVector,
                      Eigen::Vector3d &base1,
                      Eigen::Vector3d &base2);
};

#endif //STEREO_EXTRINSIC_CALIBRATION_STEREO_CALIBRATION_H
