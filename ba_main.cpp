#include <ceres/rotation.h>
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

  template <class T>
  void Evaluate(const T* camera, const T* point, T* residual) {
    T result[3];
    ceres::AngleAxisRotatePoint(camera, point, result);
    result[0] = result[0] + camera[3];
    result[1] = result[1] + camera[4];
    result[2] = result[2] + camera[5];
    T xp = -result[0] / result[2];
    T yp = -result[1] / result[2];
    T r2 = xp * xp + yp * yp;
    T distortion = 1.0 + r2 * (camera[7] + camera[8] * r2);
    T predicted_x = camera[6] * distortion * xp;
    T predicted_y = camera[6] * distortion * yp;
    residual[0] = predicted_x - x_;
    residual[1] = predicted_y - y_;
  }
};

int main() {
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
