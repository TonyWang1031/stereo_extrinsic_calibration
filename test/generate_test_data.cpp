//
// 合成测试数据生成器
// 已知GT外参，生成3D点投影到双目图像平面，加高斯噪声
//

#include <iostream>
#include <random>
#include <vector>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

int main()
{
    // GT相机内参 (与camera_params.yaml一致)
    const double fx = 500.0, fy = 500.0, cx = 320.0, cy = 240.0;
    const int width = 640, height = 480;

    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // GT外参: 小旋转 + 基线0.1m
    // 右相机相对左相机: R = Rz(1°) * Ry(0.5°), t = [0.1, 0.0, 0.0]
    double angZ = 1.0 * M_PI / 180.0;
    double angY = 0.5 * M_PI / 180.0;

    Eigen::Matrix3d Rz, Ry, R_gt;
    Rz << cos(angZ), -sin(angZ), 0,
          sin(angZ),  cos(angZ), 0,
          0,          0,         1;
    Ry << cos(angY),  0, sin(angY),
          0,          1, 0,
          -sin(angY), 0, cos(angY);
    R_gt = Rz * Ry;

    Eigen::Vector3d t_gt(0.1, 0.0, 0.0);

    std::cout << "=== Ground Truth ===" << std::endl;
    std::cout << "R:\n" << R_gt << std::endl;
    std::cout << "t: " << t_gt.transpose() << std::endl;

    // 生成随机3D点 (在两个相机前方, z=2~8m)
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> distX(-2.0, 2.0);
    std::uniform_real_distribution<double> distY(-1.5, 1.5);
    std::uniform_real_distribution<double> distZ(2.0, 8.0);
    std::normal_distribution<double> noise(0.0, 0.5); // 像素噪声

    const int numPoints = 200;

    cv::Mat imageLeft = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat imageRight = cv::Mat::zeros(height, width, CV_8UC1);

    int validCount = 0;
    for (int n = 0; n < numPoints; ++n)
    {
        Eigen::Vector3d P(distX(rng), distY(rng), distZ(rng));

        // 投影到左相机 (左相机在世界原点)
        double uL = fx * P.x() / P.z() + cx;
        double vL = fy * P.y() / P.z() + cy;

        // 投影到右相机
        Eigen::Vector3d Pr = R_gt * P + t_gt;
        if (Pr.z() <= 0) continue;
        double uR = fx * Pr.x() / Pr.z() + cx;
        double vR = fy * Pr.y() / Pr.z() + cy;

        // 加像素噪声
        uL += noise(rng); vL += noise(rng);
        uR += noise(rng); vR += noise(rng);

        // 边界检查
        int iuL = static_cast<int>(uL), ivL = static_cast<int>(vL);
        int iuR = static_cast<int>(uR), ivR = static_cast<int>(vR);
        if (iuL < 4 || iuL >= width-4 || ivL < 4 || ivL >= height-4) continue;
        if (iuR < 4 || iuR >= width-4 || ivR < 4 || ivR >= height-4) continue;

        // 在图像上绘制特征块 (7x7 高斯斑点)
        int intensity = 128 + static_cast<int>(noise(rng) * 40);
        intensity = std::max(80, std::min(255, intensity));
        cv::circle(imageLeft, cv::Point(iuL, ivL), 3, cv::Scalar(intensity), -1);
        cv::circle(imageRight, cv::Point(iuR, ivR), 3, cv::Scalar(intensity), -1);
        ++validCount;
    }

    // 添加背景纹理噪声使ORB能检测到更多特征
    cv::Mat noiseLeft(height, width, CV_8UC1), noiseRight(height, width, CV_8UC1);
    cv::randn(noiseLeft, 30, 15);
    cv::randn(noiseRight, 30, 15);
    imageLeft += noiseLeft;
    imageRight += noiseRight;

    std::string basePath = "/home/tony/claude/stereo_extrinsic_calibration/data/";
    cv::imwrite(basePath + "left.png", imageLeft);
    cv::imwrite(basePath + "right.png", imageRight);

    std::cout << "生成 " << validCount << " 个有效特征点" << std::endl;
    std::cout << "图像已保存到 " << basePath << std::endl;

    return 0;
}