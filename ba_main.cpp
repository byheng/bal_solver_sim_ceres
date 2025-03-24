// #include <ceres/rotation.h> // 使用 ceres::AngleAxisRotatePoint 时开启
#include <dirent.h>
#include <unistd.h>

#include <fstream>
#include <iostream>

#include "BoundleAdjustmentByNode_accelebrate.h"
#include "common.hpp"
#include "point_camera.h"

using namespace BoundleAdjustment;
using namespace std;

class costfunction {
 public:
  double x_;
  double y_;
  costfunction(double x, double y) : x_(x), y_(y) {}

  // 计算相机模型的重投影误差，这里使用的是针孔相机模型
  template <class T>
  void Evaluate(const T* camera, const T* point, T* residual) {

    // 根据相机参数camera旋转点point，结果保存在result中
    // 将 point 旋转到相机坐标系下
    T result[3];
    // ceres::AngleAxisRotatePoint(camera, point, result);     
    AngleAxisRotatePoint(camera, point, result);     
    
    // 平移
    result[0] = result[0] + camera[3];
    result[1] = result[1] + camera[4];
    result[2] = result[2] + camera[5];

    // 归一化
    T xp = -result[0] / result[2];
    T yp = -result[1] / result[2];

    // 畸变模型
    T r2 = xp * xp + yp * yp;
    T distortion = 1.0 + r2 * (camera[7] + camera[8] * r2);

    // 计算预测的像素坐标
    T predicted_x = camera[6] * distortion * xp;
    T predicted_y = camera[6] * distortion * yp;

    // 计算残差=预测值-观测值
    residual[0] = predicted_x - x_;
    residual[1] = predicted_y - y_;
  }

  // 计算向量x和y的点积
  template <typename T>
  inline T DotProduct(const T x[3], const T y[3]) {
    return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
  }

  // 计算旋转点point，结果保存在result中
  template <typename T>
  inline void AngleAxisRotatePoint(const T angle_axis[3], const T pt[3],
                                  T result[3]) {
    const T theta2 = DotProduct(angle_axis, angle_axis);
    if (theta2 > T(std::numeric_limits<double>::epsilon()))
    {
      // Away from zero, use the rodriguez formula
      //
      //   result = pt costheta +
      //            (w x pt) * sintheta +
      //            w (w . pt) (1 - costheta)
      //
      // We want to be careful to only evaluate the square root if the
      // norm of the angle_axis vector is greater than zero. Otherwise
      // we get a division by zero.
      //
      const T theta = sqrt(theta2);
      const T costheta = cos(theta);
      const T sintheta = sin(theta);
      const T theta_inverse = T(1.0) / theta;

      const T w[3] = {angle_axis[0] * theta_inverse,
                      angle_axis[1] * theta_inverse,
                      angle_axis[2] * theta_inverse};

      // Explicitly inlined evaluation of the cross product for
      // performance reasons.
      // (w x pt)
      const T w_cross_pt[3] = {w[1] * pt[2] - w[2] * pt[1],
                              w[2] * pt[0] - w[0] * pt[2],
                              w[0] * pt[1] - w[1] * pt[0]};
      // w . pt (1 - costheta)
      const T tmp =
          (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (T(1.0) - costheta);

      result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
      result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
      result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
    }
    else
    {
      // Near zero, the first order Taylor approximation of the rotation
      // matrix R corresponding to a vector w and angle w is
      //
      //   R = I + hat(w) * sin(theta)
      //
      // But sintheta ~ theta and theta * w = angle_axis, which gives us
      //
      //  R = I + hat(w)
      //
      // and actually performing multiplication with the point pt, gives us
      // R * pt = pt + w x pt.
      //
      // Switching to the Taylor expansion near zero provides meaningful
      // derivatives when evaluated using Jets.
      //
      // Explicitly inlined evaluation of the cross product for
      // performance reasons.
      const T w_cross_pt[3] = {angle_axis[1] * pt[2] - angle_axis[2] * pt[1],
                              angle_axis[2] * pt[0] - angle_axis[0] * pt[2],
                              angle_axis[0] * pt[1] - angle_axis[1] * pt[0]};

      result[0] = pt[0] + w_cross_pt[0];
      result[1] = pt[1] + w_cross_pt[1];
      result[2] = pt[2] + w_cross_pt[2];
    }
}
};

/*------数据集格式------
[camera number][point number][observation number] # 相机数量，点数量，观测数量
[camera id][point id][x][y] # 第1个观测数据
...
[camera id][point id][x][y] # 第observation number个观测数据
[camera parameter] # 第1组相机参数:1
...
[camera parameter] # 第1组相机参数:9
...
...
[camera parameter] # 第camera number组相机参数:9
[point parameter] # 第1组点参数:1
...
[point parameter] # 第1组点参数:3
...
...
[point parameter] # 第point number组点参数:3
*/
int main() {
  // string filename = "../data/test.txt";
  string filename = "../data/problem-16-22106-pre.txt";
  ifstream infile;
  int cn, pn, obn;  // number of camera, point, observation

  infile.open(filename);
  vector<myCamera> camera; // 相机参数，9个参数
  vector<myPoints> point; // 点参数，3个参数
  vector<myobservation> ob; // 观测，2个参数
  if (infile.is_open()) {
    infile >> cn >> pn >> obn;

    // 读取 obn 个观测数据
    for (int i = 0; i < obn; i++) {
      myobservation temp_ob;
      infile >> temp_ob.camera_id >> temp_ob.point_id >>
          temp_ob.observation[0] >> temp_ob.observation[1];
      ob.push_back(temp_ob);
    }

    // 读取 cn 个相机参数
    for (int i = 0; i < cn; i++) {
      myCamera temp_camera;
      for (int j = 0; j < 9; j++) infile >> temp_camera.parameter[j];
      camera.push_back(temp_camera);
    }
    
    // 读取 pn 个点参数
    for (int i = 0; i < pn; i++) {
      myPoints temp_point;
      for (int j = 0; j < 3; j++) {
        infile >> temp_point.parameter[j];
      }
      point.push_back(temp_point);
    }
  }

  cout << "camera number:" << cn << endl;
  cout << "point number:" << pn << endl;
  cout << "observation number:" << obn << endl;

  // 输出初始点云
  WriteToPLYFile("./initial.ply", camera, point);
  
  infile.close();

  // 定义优化问题
  Problem<2, 9, 3> problem;
  for (int i = 0; i < obn; i++) {
    // 添加观测
    Residual_node<costfunction, 2, 9, 3>* residual_node =
        new Residual_node<costfunction, 2, 9, 3>(
            new costfunction(ob[i].observation[0], ob[i].observation[1]));
    
    // 添加观测数据及其对应的相机参数和点参数
    problem.addParameterBlock(camera[ob[i].camera_id].parameter,
                              point[ob[i].point_id].parameter, 
                              residual_node);
  }
  printf("begin solve\n");
  clock_t start, finish;
  double totaltime;
  start = clock();
  problem.solve();
  finish = clock();
  totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
  cout << "\n此程序的运行时间为" << totaltime << "秒！" << endl;
  WriteToPLYFile("./final.ply", camera, point);
  return 0;
}
