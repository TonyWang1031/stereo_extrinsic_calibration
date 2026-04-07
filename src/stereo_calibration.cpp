//
// Created by tony on 04/03/26.
//

#include "/home/tony/claude/stereo_extrinsic_calibration/include/stereo_calibration.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

// 构造函数
StereoCalibration::StereoCalibration():PriorParameter(),
                                       PosteriorParameter(),
                                       CurrentRotation(Eigen::Matrix3d::Identity()),
                                       CurrentTranslation(Eigen::Vector3d::Zero()),
                                       IsInitialized(false)
{
    // 创建ORB特征检测器
    OrbDetector = cv::ORB::create(2000);

    // 创建暴力匹配器（不使用crossCheck，配合ratio test）
    BfMatcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
}

// 析构函数
StereoCalibration::~StereoCalibration()
{
}

// 特征提取与匹配
void StereoCalibration::FeatureMatching(const cv::Mat &imageLeft,
                                        const cv::Mat &imageRight,
                                        std::vector<cv::Point2f> &matchedPointsLeft,
                                        std::vector<cv::Point2f> &matchedPointsRight)
{
    // 提取特征点和描述子
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    cv::Mat descriptorsLeft, descriptorsRight;

    OrbDetector -> detectAndCompute(imageLeft, cv::noArray(), keypointsLeft, descriptorsLeft);
    OrbDetector -> detectAndCompute(imageRight, cv::noArray(), keypointsRight, descriptorsRight);

    if (descriptorsLeft.empty() || descriptorsRight.empty())
    {
        return;
    }

    // Lowe's ratio test 筛选匹配（论文Section V-A）
    // 对每个特征点找2个最近邻，要求最近邻距离远小于次近邻: d₁/d₂ < 0.75
    std::vector<std::vector<cv::DMatch>> knnMatches;
    BfMatcher -> knnMatch(descriptorsLeft, descriptorsRight, knnMatches, 2);

    matchedPointsLeft.clear();
    matchedPointsRight.clear();

    // 第一级筛选: Lowe's ratio test（论文Section V-A）
    // 对每个特征点找2个最近邻，要求最近邻距离远小于次近邻: d₁/d₂ < 0.75
    const float RatioThreshold = 0.75f;                                            ///< Lowe's ratio test阈值
    std::vector<cv::DMatch> ratioPassedMatches;
    for (const auto &knnMatch : knnMatches)
    {
        if (knnMatch.size() >= 2 && knnMatch[0].distance < RatioThreshold * knnMatch[1].distance)
        {
            ratioPassedMatches.push_back(knnMatch[0]);
        }
    }

    // 第二级筛选: 距离阈值（论文Section V-A）
    // dist < max(2 × min_distance, 30)
    float minDist = std::numeric_limits<float>::max();
    for (const auto &match : ratioPassedMatches)
    {
        if (match.distance < minDist)
        {
            minDist = match.distance;
        }
    }

    const float DistanceThreshold = std::max(2.0f * minDist, 30.0f);
    for (const auto &match : ratioPassedMatches)
    {
        if (match.distance < DistanceThreshold)
        {
            matchedPointsLeft.push_back(keypointsLeft[match.queryIdx].pt);
            matchedPointsRight.push_back(keypointsRight[match.trainIdx].pt);
        }
    }
}

// 本质矩阵估计初始位姿（支持左右内参不同）
bool StereoCalibration::EstimateInitialPose(const std::vector<cv::Point2f> &matchedPointsLeft,
                                            const std::vector<cv::Point2f> &matchedPointsRight,
                                            const Eigen::Matrix3d &internalMatrixLeft,
                                            const Eigen::Matrix3d &internalMatrixRight,
                                            const cv::Mat &distortionCoeffsLeft,
                                            const cv::Mat &distortionCoeffsRight,
                                            Eigen::Matrix3d &rotationMatrix,
                                            Eigen::Vector3d &translationVector)
{
    if (matchedPointsLeft.size() < 8)
    {
        return false;
    }

    // 用各自内参归一化匹配点: f = K⁻¹ · [u, v, 1]ᵀ
    // 归一化后用单位矩阵作为内参调用findEssentialMat，避免假设左右内参相同
    cv::Mat cvInternalMatrixLeft, cvInternalMatrixRight;
    cv::eigen2cv(internalMatrixLeft, cvInternalMatrixLeft);
    cv::eigen2cv(internalMatrixRight, cvInternalMatrixRight);

    std::vector<cv::Point2f> normalizedLeft, normalizedRight;
    cv::undistortPoints(matchedPointsLeft, normalizedLeft, cvInternalMatrixLeft, distortionCoeffsLeft);
    cv::undistortPoints(matchedPointsRight, normalizedRight, cvInternalMatrixRight, distortionCoeffsRight);

    // 归一化坐标下用单位矩阵计算本质矩阵
    cv::Mat identityK = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat essentialMatrix = cv::findEssentialMat(normalizedLeft,
                                                    normalizedRight,
                                                    identityK,
                                                    cv::RANSAC,
                                                    0.999,
                                                    1e-3);

    if (essentialMatrix.empty())
    {
        return false;
    }

    // 从本质矩阵恢复R和t
    cv::Mat cvRotationMatrix, cvTranslationVector, mask;
    int inliers = cv::recoverPose(essentialMatrix,
                                   normalizedLeft,
                                   normalizedRight,
                                   identityK,
                                   cvRotationMatrix,
                                   cvTranslationVector,
                                   mask);

    if (inliers < 8)
    {
        return false;
    }

    // 转换为Eigen格式
    cv::cv2eigen(cvRotationMatrix, rotationMatrix);
    cv::cv2eigen(cvTranslationVector, translationVector);

    return true;
}

// 三角化重建3D点
void StereoCalibration::TriangulatePoints(const std::vector<cv::Point2f> &matchedPointsLeft,
                                          const std::vector<cv::Point2f> &matchedPointsRight,
                                          const Eigen::Matrix3d &internalMatrixLeft,
                                          const Eigen::Matrix3d &internalMatrixRight,
                                          const Eigen::Matrix3d &rotationMatrix,
                                          const Eigen::Vector3d &translationVector,
                                          std::vector<Eigen::Vector3d> &points3D)
{
    // 转换为OpenCV格式
    cv::Mat cvInternalMatrixLeft, cvInternalMatrixRight;
    cv::eigen2cv(internalMatrixLeft, cvInternalMatrixLeft);
    cv::eigen2cv(internalMatrixRight, cvInternalMatrixRight);

    // 构建投影矩阵 P_left = K_left * [I | 0]
    cv::Mat projectionMatrixLeft = cv::Mat::zeros(3, 4, CV_64F);
    cvInternalMatrixLeft.copyTo(projectionMatrixLeft(cv::Rect(0, 0, 3, 3)));

    // 构建投影矩阵 P_right = K_right * [R | t]
    cv::Mat cvRotationMatrix, cvTranslationVector;
    cv::eigen2cv(rotationMatrix, cvRotationMatrix);
    cv::eigen2cv(translationVector, cvTranslationVector);

    cv::Mat rotationTranslation;
    cv::hconcat(cvRotationMatrix, cvTranslationVector, rotationTranslation);
    cv::Mat projectionMatrixRight = cvInternalMatrixRight * rotationTranslation;

    // 三角化
    cv::Mat points4D;
    cv::triangulatePoints(projectionMatrixLeft, projectionMatrixRight, matchedPointsLeft, matchedPointsRight, points4D);

    // 齐次坐标转3D坐标
    points3D.clear();
    for (int i = 0; i < points4D.cols; ++i)
    {
        float w = points4D.at<float>(3, i);
        if (std::abs(w) < 1e-6)
        {
            continue;
        }

        Eigen::Vector3d point3D;
        point3D.x() = points4D.at<float>(0, i) / w;
        point3D.y() = points4D.at<float>(1, i) / w;
        point3D.z() = points4D.at<float>(2, i) / w;

        // 深度检查：只保留在两个相机前方的点
        if (point3D.z() > 0)
        {
            Eigen::Vector3d pointRight = rotationMatrix * point3D + translationVector;
            if (pointRight.z() > 0)
            {
                points3D.push_back(point3D);
            }
        }
    }
}

// 构造反对称矩阵 [v]×
// 输入向量 v = [v₁, v₂, v₃]ᵀ，输出:
// [  0  -v₃  v₂ ]
// [  v₃  0  -v₁ ]
// [ -v₂  v₁  0  ]
Eigen::Matrix3d StereoCalibration::SkewSymmetric(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d skew;
    skew <<     0, -v(2),  v(1),
             v(2),     0, -v(0),
            -v(1),  v(0),     0;
    return skew;
}

// 指数映射 exp([δθ]×)（Rodrigues公式）
// exp([δθ]×) = I + sin(θ)/θ · [δθ]× + (1-cos(θ))/θ² · [δθ]×²
// 其中 θ = ||δθ||
Eigen::Matrix3d StereoCalibration::ExpMap(const Eigen::Vector3d &deltaTheta)
{
    double theta = deltaTheta.norm();

    if (theta < 1e-10)
    {
        // 小角度近似: exp([δθ]×) ≈ I + [δθ]×
        return Eigen::Matrix3d::Identity() + SkewSymmetric(deltaTheta);
    }

    Eigen::Matrix3d skew = SkewSymmetric(deltaTheta);
    double sinTheta = std::sin(theta);
    double cosTheta = std::cos(theta);

    // Rodrigues公式
    return Eigen::Matrix3d::Identity()
           + (sinTheta / theta) * skew
           + ((1.0 - cosTheta) / (theta * theta)) * skew * skew;
}

// 构造切空间正交基（论文Algorithm 2: FindingBases）
// 在单位球面 S² 上点 t 处，构造切平面的两个正交单位基向量 b₁, b₂
// 步骤:
//   1. 找到 t 中绝对值最小的分量索引 i
//   2. 构造初始向量 v（第 i 个分量为1，其余为0）
//   3. b₁ = normalize(v - (vᵀt)t)  （Gram-Schmidt正交化）
//   4. b₂ = t × b₁                  （叉积得到第二个基）
void StereoCalibration::FindingBases(const Eigen::Vector3d &translationVector,
                                     Eigen::Vector3d &base1,
                                     Eigen::Vector3d &base2)
{
    // 找到绝对值最小的分量索引（避免数值不稳定）
    int minIndex = 0;
    double minValue = std::abs(translationVector(0));
    for (int i = 1; i < 3; ++i)
    {
        if (std::abs(translationVector(i)) < minValue)
        {
            minValue = std::abs(translationVector(i));
            minIndex = i;
        }
    }

    // 构造初始向量 v = eᵢ（第 minIndex 个标准基向量）
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    v(minIndex) = 1.0;

    // Gram-Schmidt正交化: b₁ = normalize(v - (vᵀt)t)
    base1 = v - v.dot(translationVector) * translationVector;
    base1.normalize();

    // 叉积得到第二个基: b₂ = t × b₁
    base2 = translationVector.cross(base1);
    base2.normalize();
}

// 5自由度非线性优化（论文Algorithm 1: NonlinearOptimization）
// 优化变量: ξ = [δθ₁, δθ₂, δθ₃, α, β]ᵀ ∈ ℝ⁵
//   - δθ ∈ ℝ³: 旋转增量（李代数 so(3)）
//   - [α, β]ᵀ ∈ ℝ²: 平移切空间增量
// 代价函数: C(ξ) = Σᵢ rᵢ²，其中 rᵢ = (f'ᵢ)ᵀ · E · fᵢ（极线误差）
// 迭代步骤:
//   1. 构造切空间基 B = [b₁, b₂]（Algorithm 2）
//   2. 计算本质矩阵 E = [t]× · R
//   3. 计算残差 r 和雅可比 J（公式12-14）
//   4. 求解正规方程 (JᵀJ)·Δξ = -Jᵀr
//   5. 更新: R ← R·exp([δθ]×), t ← normalize(t + B·δt)
void StereoCalibration::NonlinearOptimization(
    const std::vector<Eigen::Vector3d> &normalizedPointsLeft,
    const std::vector<Eigen::Vector3d> &normalizedPointsRight,
    Eigen::Matrix3d &rotationMatrix,
    Eigen::Vector3d &translationVector,
    double &epipolarError)
{
    const int MaxIterations = 20;                                               ///< 最大迭代次数
    const double ConvergenceThreshold = 1e-6;                                   ///< 增量收敛阈值 ||Δξ|| < ε₁
    const double CostChangeThreshold = 1e-8;                                    ///< 代价变化阈值 |ΔC| < ε₂

    int numPoints = static_cast<int>(normalizedPointsLeft.size());

    // 归一化平移向量为单位向量（||t|| = 1 约束）
    translationVector.normalize();

    double previousCost = std::numeric_limits<double>::max();

    for (int iter = 0; iter < MaxIterations; ++iter)
    {
        // Step 1: 构造切空间正交基 B = [b₁, b₂]（Algorithm 2）
        Eigen::Vector3d base1, base2;
        FindingBases(translationVector, base1, base2);

        // Step 2: 构造残差向量 r ∈ ℝᴺ 和雅可比矩阵 J ∈ ℝᴺˣ⁵
        Eigen::VectorXd residuals(numPoints);
        Eigen::MatrixXd jacobian(numPoints, 5);

        for (int i = 0; i < numPoints; ++i)
        {
            const Eigen::Vector3d &fl = normalizedPointsLeft[i];               ///< 左归一化坐标 fᵢ
            const Eigen::Vector3d &fr = normalizedPointsRight[i];              ///< 右归一化坐标 f'ᵢ

            // 本质矩阵 E = [t]× · R
            Eigen::Matrix3d E = SkewSymmetric(translationVector) * rotationMatrix;

            // Sampson距离: r_s = (f'^T E f) / sqrt((Ef)_0^2 + (Ef)_1^2 + (E^Tf')_0^2 + (E^Tf')_1^2)
            Eigen::Vector3d Ef = E * fl;
            Eigen::Vector3d Etfr = E.transpose() * fr;
            double num = fr.dot(E * fl);
            double denom = std::sqrt(Ef(0)*Ef(0) + Ef(1)*Ef(1) + Etfr(0)*Etfr(0) + Etfr(1)*Etfr(1));

            if (denom < 1e-12) denom = 1e-12;
            residuals(i) = num / denom;

            // 旋转后的左归一化点: Rfᵢ = R · fᵢ
            Eigen::Vector3d Rfl = rotationMatrix * fl;

            // --- Sampson距离的雅可比 (商法则) ---
            // J_algebraic: 代数误差 num = f'^T E f 对 ξ 的雅可比
            // 旋转部分
            Eigen::Vector3d w = -rotationMatrix.transpose() * translationVector.cross(fr);
            Eigen::Vector3d jRotAlg = fl.cross(w);
            // 平移部分
            Eigen::Vector3d c = Rfl.cross(fr);
            Eigen::Matrix<double, 1, 5> J_num;
            J_num << jRotAlg(0), jRotAlg(1), jRotAlg(2), c.dot(base1), c.dot(base2);

            // J_denom: denom 对 ξ 的雅可比
            // denom = sqrt(s), s = Ef_0^2 + Ef_1^2 + Etfr_0^2 + Etfr_1^2
            // d(denom)/dξ = (1/(2*denom)) * d(s)/dξ
            // d(Ef)/dξ_rot_j = [t]× R [e_j]× f = SkewSymmetric(t) * R * SkewSymmetric(e_j) * f
            // d(Ef)/dξ_trans_j = [b_j]× R f
            Eigen::Matrix<double, 3, 5> dEf, dEtfr;
            for (int j = 0; j < 3; ++j)
            {
                Eigen::Vector3d ej = Eigen::Vector3d::Zero();
                ej(j) = 1.0;
                // d(E*fl)/d(delta_theta_j) = [t]× * R * [ej]× * fl
                dEf.col(j) = SkewSymmetric(translationVector) * rotationMatrix * SkewSymmetric(ej) * fl;
                // d(E^T*fr)/d(delta_theta_j) = ([t]× * R * [ej]×)^T * fr = -[ej]× * R^T * [t]× * fr
                dEtfr.col(j) = (SkewSymmetric(translationVector) * rotationMatrix * SkewSymmetric(ej)).transpose() * fr;
            }
            // 平移部分
            dEf.col(3) = SkewSymmetric(base1) * rotationMatrix * fl;
            dEf.col(4) = SkewSymmetric(base2) * rotationMatrix * fl;
            dEtfr.col(3) = (SkewSymmetric(base1) * rotationMatrix).transpose() * fr;
            dEtfr.col(4) = (SkewSymmetric(base2) * rotationMatrix).transpose() * fr;

            // ds/dξ = 2*(Ef_0*dEf_0 + Ef_1*dEf_1 + Etfr_0*dEtfr_0 + Etfr_1*dEtfr_1)
            Eigen::Matrix<double, 1, 5> ds;
            ds = 2.0 * (Ef(0) * dEf.row(0) + Ef(1) * dEf.row(1) + Etfr(0) * dEtfr.row(0) + Etfr(1) * dEtfr.row(1));
            Eigen::Matrix<double, 1, 5> J_denom = ds / (2.0 * denom);

            // 商法则: J_sampson = (J_num * denom - num * J_denom) / denom^2
            jacobian.row(i) = (J_num * denom - num * J_denom) / (denom * denom);
        }

        // Step 3: 收敛判断 — 代价函数变化
        double currentCost = residuals.squaredNorm();
        if (std::abs(previousCost - currentCost) < CostChangeThreshold)
        {
            std::cout << "5-DOF优化收敛（代价变化），迭代次数: " << iter << std::endl;
            break;
        }
        previousCost = currentCost;

        // Step 4: 求解正规方程 (JᵀJ)·Δξ = -Jᵀr
        // JᵀJ 是 5×5 对称正定矩阵，用LDLT分解高效求解
        Eigen::Matrix<double, 5, 5> JtJ = jacobian.transpose() * jacobian;
        Eigen::Matrix<double, 5, 1> Jtr = jacobian.transpose() * residuals;
        Eigen::Matrix<double, 5, 1> deltaXi = -JtJ.ldlt().solve(Jtr);

        // Step 5: 收敛判断 — 增量足够小
        if (deltaXi.norm() < ConvergenceThreshold)
        {
            std::cout << "5-DOF优化收敛（增量阈值），迭代次数: " << iter << std::endl;
            break;
        }

        // Step 6: 更新旋转 R ← R · exp([δθ]×)
        Eigen::Vector3d deltaTheta = deltaXi.head<3>();
        rotationMatrix = rotationMatrix * ExpMap(deltaTheta);

        // Step 7: 更新平移 t ← normalize(t + B·δt)
        // 在切空间上做小扰动后重新投影到单位球面
        Eigen::Vector2d deltaTrans = deltaXi.tail<2>();
        translationVector = translationVector + base1 * deltaTrans(0) + base2 * deltaTrans(1);
        translationVector.normalize();
    }

    // 计算最终的RMS Sampson距离
    Eigen::Matrix3d finalE = SkewSymmetric(translationVector) * rotationMatrix;
    double totalError = 0.0;
    for (int i = 0; i < numPoints; ++i)
    {
        Eigen::Vector3d Ef = finalE * normalizedPointsLeft[i];
        Eigen::Vector3d Etfr = finalE.transpose() * normalizedPointsRight[i];
        double num = normalizedPointsRight[i].transpose() * finalE * normalizedPointsLeft[i];
        double denom = std::sqrt(Ef(0)*Ef(0) + Ef(1)*Ef(1) + Etfr(0)*Etfr(0) + Etfr(1)*Etfr(1));
        if (denom < 1e-12) denom = 1e-12;
        double sampson = num / denom;
        totalError += sampson * sampson;
    }
    epipolarError = std::sqrt(totalError / numPoints);
    std::cout << "5-DOF优化完成，RMS Sampson距离: " << epipolarError << std::endl;
}

// RANSAC外点剔除 + 5-DOF优化（论文Section V-B）
// 流程:
//   1. 随机采样最小集（5对点）
//   2. 用最小集做5-DOF优化得到候选模型
//   3. 计算所有点的极线误差，统计内点
//   4. 保留内点最多的模型
//   5. 用全部内点重新做5-DOF优化
void StereoCalibration::RansacOptimization(
    const std::vector<Eigen::Vector3d> &normalizedPointsLeft,
    const std::vector<Eigen::Vector3d> &normalizedPointsRight,
    Eigen::Matrix3d &rotationMatrix,
    Eigen::Vector3d &translationVector,
    double &epipolarError,
    std::vector<int> &inlierIndices)
{
    const int MinSampleSize = 5;                                                   ///< 最小采样集大小
    const int HardMaxIterations = 300;                                             ///< RANSAC硬上限迭代次数
    const double InlierThreshold = 1e-3;                                           ///< Sampson距离内点阈值
    const double DesiredConfidence = 0.999;                                        ///< 期望置信度

    int numPoints = static_cast<int>(normalizedPointsLeft.size());
    if (numPoints < MinSampleSize)
    {
        return;
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, numPoints - 1);

    int bestInlierCount = 0;
    Eigen::Matrix3d bestRotation = rotationMatrix;
    Eigen::Vector3d bestTranslation = translationVector;
    int adaptiveMaxIter = HardMaxIterations;

    for (int iter = 0; iter < adaptiveMaxIter; ++iter)
    {
        // Step 1: 随机采样最小集
        std::vector<int> sampleIndices;
        while (static_cast<int>(sampleIndices.size()) < MinSampleSize)
        {
            int idx = dist(rng);
            bool duplicate = false;
            for (int si : sampleIndices)
            {
                if (si == idx) { duplicate = true; break; }
            }
            if (!duplicate)
            {
                sampleIndices.push_back(idx);
            }
        }

        std::vector<Eigen::Vector3d> sampleLeft, sampleRight;
        for (int idx : sampleIndices)
        {
            sampleLeft.push_back(normalizedPointsLeft[idx]);
            sampleRight.push_back(normalizedPointsRight[idx]);
        }

        // Step 2: 用最小集做5-DOF优化
        Eigen::Matrix3d candidateR = rotationMatrix;
        Eigen::Vector3d candidateT = translationVector;
        double candidateError = 0.0;
        NonlinearOptimization(sampleLeft, sampleRight, candidateR, candidateT, candidateError);

        // Step 3: 计算所有点的Sampson距离，统计内点
        Eigen::Matrix3d E = SkewSymmetric(candidateT) * candidateR;
        int inlierCount = 0;
        for (int i = 0; i < numPoints; ++i)
        {
            Eigen::Vector3d Ef = E * normalizedPointsLeft[i];
            Eigen::Vector3d Etfr = E.transpose() * normalizedPointsRight[i];
            double num = normalizedPointsRight[i].transpose() * E * normalizedPointsLeft[i];
            double denom = std::sqrt(Ef(0)*Ef(0) + Ef(1)*Ef(1) + Etfr(0)*Etfr(0) + Etfr(1)*Etfr(1));
            if (denom < 1e-12) denom = 1e-12;
            double sampson = std::abs(num / denom);
            if (sampson < InlierThreshold)
            {
                ++inlierCount;
            }
        }

        // Step 4: 保留内点最多的模型，自适应更新迭代上限
        if (inlierCount > bestInlierCount)
        {
            bestInlierCount = inlierCount;
            bestRotation = candidateR;
            bestTranslation = candidateT;

            // 自适应RANSAC: maxIter = log(1-p) / log(1-w^s)
            double w = static_cast<double>(inlierCount) / numPoints;
            double wPow = std::pow(w, MinSampleSize);
            if (wPow > 1.0 - 1e-10) wPow = 1.0 - 1e-10;
            int newMaxIter = static_cast<int>(std::ceil(std::log(1.0 - DesiredConfidence) / std::log(1.0 - wPow)));
            adaptiveMaxIter = std::min(newMaxIter, HardMaxIterations);
        }
    }

    // Step 5: 收集最优模型的内点（Sampson距离）
    Eigen::Matrix3d bestE = SkewSymmetric(bestTranslation) * bestRotation;
    std::vector<Eigen::Vector3d> inlierLeft, inlierRight;
    inlierIndices.clear();
    for (int i = 0; i < numPoints; ++i)
    {
        Eigen::Vector3d Ef = bestE * normalizedPointsLeft[i];
        Eigen::Vector3d Etfr = bestE.transpose() * normalizedPointsRight[i];
        double num = normalizedPointsRight[i].transpose() * bestE * normalizedPointsLeft[i];
        double denom = std::sqrt(Ef(0)*Ef(0) + Ef(1)*Ef(1) + Etfr(0)*Etfr(0) + Etfr(1)*Etfr(1));
        if (denom < 1e-12) denom = 1e-12;
        double sampson = std::abs(num / denom);
        if (sampson < InlierThreshold)
        {
            inlierLeft.push_back(normalizedPointsLeft[i]);
            inlierRight.push_back(normalizedPointsRight[i]);
            inlierIndices.push_back(i);
        }
    }

    std::cout << "RANSAC: " << bestInlierCount << "/" << numPoints << " 内点" << std::endl;

    // Step 6: 用全部内点重新做5-DOF优化
    rotationMatrix = bestRotation;
    translationVector = bestTranslation;
    NonlinearOptimization(inlierLeft, inlierRight, rotationMatrix, translationVector, epipolarError);
}

// 累积帧到滑动窗口（论文Section V-C）
// 维护最近 MaxWindowSize 帧的归一化匹配点
void StereoCalibration::AccumulateFrame(
    const std::vector<Eigen::Vector3d> &normalizedPointsLeft,
    const std::vector<Eigen::Vector3d> &normalizedPointsRight)
{
    FrameBufferLeft.push_back(normalizedPointsLeft);
    FrameBufferRight.push_back(normalizedPointsRight);

    // 超过窗口大小时移除最旧的帧
    if (static_cast<int>(FrameBufferLeft.size()) > MaxWindowSize)
    {
        FrameBufferLeft.erase(FrameBufferLeft.begin());
        FrameBufferRight.erase(FrameBufferRight.begin());
    }
}

// 多帧融合标定（论文Section V-C）
// 将滑动窗口内所有帧的归一化点合并，联合优化: C = Σₖ Σᵢ ||rᵢᵏ||²
void StereoCalibration::CalibrateMultiFrame()
{
    if (FrameBufferLeft.empty())
    {
        return;
    }

    // Warm start: 如果尚未初始化，用第一帧的点做初始位姿估计
    if (!IsInitialized && !FrameBufferLeft[0].empty())
    {
        // 归一化点转为像素坐标用于EstimateInitialPose
        // 这里直接用归一化点作为"像素坐标"，配合单位矩阵内参
        std::vector<cv::Point2f> initLeft, initRight;
        for (size_t i = 0; i < FrameBufferLeft[0].size(); ++i)
        {
            initLeft.push_back(cv::Point2f(FrameBufferLeft[0][i](0), FrameBufferLeft[0][i](1)));
            initRight.push_back(cv::Point2f(FrameBufferRight[0][i](0), FrameBufferRight[0][i](1)));
        }
        Eigen::Matrix3d identityK = Eigen::Matrix3d::Identity();
        cv::Mat zeroDist = cv::Mat::zeros(1, 5, CV_64F);
        EstimateInitialPose(initLeft, initRight, identityK, identityK,
                            zeroDist, zeroDist, CurrentRotation, CurrentTranslation);
        IsInitialized = true;
    }

    // 合并所有帧的归一化点
    std::vector<Eigen::Vector3d> allPointsLeft, allPointsRight;
    for (size_t k = 0; k < FrameBufferLeft.size(); ++k)
    {
        allPointsLeft.insert(allPointsLeft.end(),
                             FrameBufferLeft[k].begin(), FrameBufferLeft[k].end());
        allPointsRight.insert(allPointsRight.end(),
                              FrameBufferRight[k].begin(), FrameBufferRight[k].end());
    }

    std::cout << "多帧融合: " << FrameBufferLeft.size() << " 帧, "
              << allPointsLeft.size() << " 个点" << std::endl;

    // RANSAC + 5-DOF联合优化
    double epipolarError = 0.0;
    std::vector<int> inlierIndices;
    RansacOptimization(allPointsLeft, allPointsRight,
                       CurrentRotation, CurrentTranslation,
                       epipolarError, inlierIndices);

    // 保存结果
    Eigen::Isometry3d posteriorExternalMatrixRight = Eigen::Isometry3d::Identity();
    posteriorExternalMatrixRight.rotate(CurrentRotation);
    posteriorExternalMatrixRight.pretranslate(CurrentTranslation);
    PosteriorParameter.SaveCalibrationResult(posteriorExternalMatrixRight, epipolarError);
}

// 设置先验参数函数
void StereoCalibration::SetPriorParameter(StereoCalibrationParam &priorParameter)
{
    PriorParameter = priorParameter;
}

// 得到后验参数函数
void StereoCalibration::GetPosteriorParameter(StereoCalibrationParam &posteriorParameter)
{
    posteriorParameter = PosteriorParameter;
}

// 执行标定
void StereoCalibration::Calibrate()
{
    // 获取参数
    Eigen::Matrix3d internalMatrixLeft, internalMatrixRight;
    PriorParameter.GetInternalMatrix(internalMatrixLeft, internalMatrixRight);

    cv::Mat distortionCoeffsLeft, distortionCoeffsRight;
    PriorParameter.GetDistortionCoeffs(distortionCoeffsLeft, distortionCoeffsRight);

    std::vector<cv::Point2f> matchedPointsLeft, matchedPointsRight;
    PriorParameter.GetMatchedPoints(matchedPointsLeft, matchedPointsRight);

    // 估计初始位姿（本质矩阵分解）
    Eigen::Matrix3d rotationMatrix;
    Eigen::Vector3d translationVector;
    if (!EstimateInitialPose(matchedPointsLeft, matchedPointsRight, internalMatrixLeft,
                             internalMatrixRight, distortionCoeffsLeft, distortionCoeffsRight,
                             rotationMatrix, translationVector))
    {
        std::cout << "初始位姿估计失败" << std::endl;
        return;
    }

    // 将像素坐标转换为归一化坐标（含畸变校正）: undistortPoints 同时去畸变和归一化
    cv::Mat cvInternalMatrixLeft, cvInternalMatrixRight;
    cv::eigen2cv(internalMatrixLeft, cvInternalMatrixLeft);
    cv::eigen2cv(internalMatrixRight, cvInternalMatrixRight);

    std::vector<cv::Point2f> undistortedLeft, undistortedRight;
    cv::undistortPoints(matchedPointsLeft, undistortedLeft, cvInternalMatrixLeft, distortionCoeffsLeft);
    cv::undistortPoints(matchedPointsRight, undistortedRight, cvInternalMatrixRight, distortionCoeffsRight);

    std::vector<Eigen::Vector3d> normalizedPointsLeft, normalizedPointsRight;
    for (size_t i = 0; i < undistortedLeft.size(); ++i)
    {
        normalizedPointsLeft.push_back(Eigen::Vector3d(undistortedLeft[i].x, undistortedLeft[i].y, 1.0));
        normalizedPointsRight.push_back(Eigen::Vector3d(undistortedRight[i].x, undistortedRight[i].y, 1.0));
    }

    // RANSAC + 5自由度非线性优化（论文核心算法）
    double epipolarError = 0.0;
    std::vector<int> inlierIndices;
    RansacOptimization(normalizedPointsLeft, normalizedPointsRight,
                       rotationMatrix, translationVector, epipolarError, inlierIndices);

    // 用内点的像素坐标做三角化（用于尺度恢复和重投影误差评估）
    std::vector<cv::Point2f> inlierPixelLeft, inlierPixelRight;
    for (int idx : inlierIndices)
    {
        inlierPixelLeft.push_back(matchedPointsLeft[idx]);
        inlierPixelRight.push_back(matchedPointsRight[idx]);
    }

    std::vector<Eigen::Vector3d> points3D;
    TriangulatePoints(inlierPixelLeft, inlierPixelRight, internalMatrixLeft,
                      internalMatrixRight, rotationMatrix, translationVector, points3D);

    // 尺度恢复: 利用先验基线长度（从配置的外参中获取）
    // 5-DOF优化输出的t是单位向量，需要恢复真实尺度
    Eigen::Isometry3d priorExternalLeft, priorExternalRight;
    PriorParameter.GetExternalMatrix(priorExternalLeft, priorExternalRight);
    Eigen::Vector3d priorTranslation = priorExternalRight.translation();
    double priorBaseline = priorTranslation.norm();

    if (priorBaseline > 1e-6)
    {
        // 有基线先验: 直接用先验基线长度缩放单位平移向量
        translationVector = translationVector * priorBaseline;
        std::cout << "尺度恢复: 基线长度 = " << priorBaseline << std::endl;
    }

    // 计算重投影误差（使用恢复尺度后的t）
    double reprojectionError = 0.0;
    if (!points3D.empty() && priorBaseline > 1e-6)
    {
        // 用恢复尺度后的参数重新三角化
        std::vector<Eigen::Vector3d> scaledPoints3D;
        TriangulatePoints(inlierPixelLeft, inlierPixelRight, internalMatrixLeft,
                          internalMatrixRight, rotationMatrix, translationVector, scaledPoints3D);

        if (!scaledPoints3D.empty())
        {
            cv::Mat cvInternalMatrixRight;
            cv::eigen2cv(internalMatrixRight, cvInternalMatrixRight);
            cv::Mat cvR, cvT, rvec;
            cv::eigen2cv(rotationMatrix, cvR);
            cv::eigen2cv(translationVector, cvT);
            cv::Rodrigues(cvR, rvec);

            std::vector<cv::Point3f> cvPoints3D;
            for (const auto &p : scaledPoints3D)
            {
                cvPoints3D.push_back(cv::Point3f(p.x(), p.y(), p.z()));
            }

            std::vector<cv::Point2f> reprojected;
            cv::projectPoints(cvPoints3D, rvec, cvT, cvInternalMatrixRight, cv::noArray(), reprojected);

            double totalError = 0.0;
            int count = std::min(reprojected.size(), inlierPixelRight.size());
            for (int i = 0; i < count; ++i)
            {
                double dx = reprojected[i].x - inlierPixelRight[i].x;
                double dy = reprojected[i].y - inlierPixelRight[i].y;
                totalError += std::sqrt(dx * dx + dy * dy);
            }
            reprojectionError = totalError / count;
        }
    }

    std::cout << "极线误差(RMS): " << epipolarError << std::endl;
    std::cout << "重投影误差: " << reprojectionError << " 像素" << std::endl;

    // 保存结果
    Eigen::Isometry3d posteriorExternalMatrixRight = Eigen::Isometry3d::Identity();
    posteriorExternalMatrixRight.rotate(rotationMatrix);
    posteriorExternalMatrixRight.pretranslate(translationVector);

    PosteriorParameter.SaveCalibrationResult(posteriorExternalMatrixRight, epipolarError);

    // 更新当前估计（供多帧融合使用）
    CurrentRotation = rotationMatrix;
    CurrentTranslation = translationVector;
    CurrentTranslation.normalize();
    IsInitialized = true;

    // 累积当前帧到滑动窗口
    std::vector<Eigen::Vector3d> inlierNormLeft, inlierNormRight;
    for (int idx : inlierIndices)
    {
        inlierNormLeft.push_back(normalizedPointsLeft[idx]);
        inlierNormRight.push_back(normalizedPointsRight[idx]);
    }
    AccumulateFrame(inlierNormLeft, inlierNormRight);
}
