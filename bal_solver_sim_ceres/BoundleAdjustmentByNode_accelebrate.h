#include <math.h>
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <iomanip>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "Eigen/Eigen"
#include "Eigen/SparseQR"
#include "algorithm"
#include "jet.h"
#include "point_camera.h"
#include "stdlib.h"
#define MAXSTEP 1e8
#define MINSTEP 1e-9
#define MINDIF 1e-6
#define PARAMETERMIN 1e-8
#define max_consecutive_nonmonotonic_steps 0
#define OUTTER_ITERATION_NUMBER 50
#define INNER_ITERATION_NUMBER 50
#define INITMIU 1e4
using namespace std;
/* The class Problem used to solve a boundleAdjustment problem.For faster the
 * compute progress,use a template to init the matrix in Problem. Because the
 * static matrix can compute faster than dynamic matrix.Notice the schur matrix
 * S could not known before the problem begin to solve.So the Schur_A and
 * Schur_B must be dynamic. The solve process can be this: First use function
 * addParameterBlock to add the parameter_camera and parameter_point to this
 * problem. Then in function solve ,we compute the residual and jacobi first and
 * use function  pre_process() to set the schur matrix's size and build a right
 * pair of camera.then init_scaling can init the scaling params once. These
 * three functions only compute once.Then do the LM loop method: First scaling
 * the jacobi matrix,then compute the hessian and write camera's hessian to
 * Schur Matrix.Also the camera's Residual(=Ji^T*Ui). Then do the
 * schur_complement to compute the delta Camera After that we get the delta step
 * of camera,then use (deltaP) = hessianP^-1*(Vi - W^T*deltaC) -->delta step of
 * Point. Compute the model_cost and then the real cost.If the
 * relative_decrease>0.001,we get a success step,update this and continue the
 * process until limit reached.
 *
 */
namespace BoundleAdjustment {
/*
  N : the dimension of the residual
  N1: the dimension of the camera parameter
  N2: the dimension of the point parameter
*/
template <int N, int N1, int N2>
class Problem
{
 public:
  Problem();
  ~Problem();
  struct Residual_block
  {
    Residual_block(int a, int b, // 相机参数和点参数的索引
                   BoundleAdjustment::CostFunction<N, N1, N2> *node) // 代价函数，包含观测值
        : camera_index(a), point_index(b)
    {
      jacobi_parameter_camera.setZero();
      jacobi_parameter_point.setZero();
      hessian_W.setZero();
      residual.setZero();
      residual_node = node;
    }
    int camera_index; // 相机参数的索引
    int point_index; // 点参数的索引
    /*The residual_node use to make the costfunction feasible,CostFunction is a
     * virtual class and its sub class Residual_node is a template class which
     * has a class T.Then class T can be writen in main and offer the key
     * function(Evaluate) to compute the residual and jacobian. Virtual class is
     * used for this trick even it consume some compute time.
     */
    BoundleAdjustment::CostFunction<N, N1, N2> *residual_node;
    Eigen::Matrix<double, N, N1> jacobi_parameter_camera;
    Eigen::Matrix<double, N, N2> jacobi_parameter_point;
    Eigen::Matrix<double, N1, N2> hessian_W; // W 矩阵=JcT*Jp
    Eigen::Matrix<double, N, 1> residual;
  };

  template <int n>
  struct Parameter
  {
    Parameter()
    {
      jacobi_scaling.setZero();
      hessian.setZero();
      hessian_inverse.setZero();
      params.setZero();
      candidate.setZero();
      delta.setZero();
      residual.setZero();
    }
    Eigen::Matrix<double, n, 1> params; // 当前参数值
    Eigen::Matrix<double, n, 1> candidate; // 候选参数值
    Eigen::Matrix<double, n, 1> delta; // 相机参数或3D点参数的变化量：deltaPc 或 deltaPp
    Eigen::Matrix<double, n, 1> residual; // 向量r的第二项的后部：JpT * e（或JcT * e）(e 为残差）    // TODO: 该变量名需要修改，应为增量 g=JT*e
    Eigen::Matrix<double, n, 1> jacobi_scaling; // Jacobian 矩阵的缩放因子
    Eigen::Matrix<double, n, n> hessian; // Hessian 矩阵=JcT*Jc(或JpT*Jp)
    // The hessian_inverse used just for time saving.
    Eigen::Matrix<double, n, n> hessian_inverse; // Hessian 矩阵的逆矩阵，用于节省时间
  };

  vector<Residual_block *> residual_block_vector; // 残差块，与观测值数量相等
  vector<Parameter<N1> *> parameter_camera_vector; // 相机参数
  vector<Parameter<N2> *> parameter_point_vector; // 点参数
  map<double *, int> parameter_camera_map;
  map<double *, int> parameter_point_map;
  int parameter_a_size;
  Eigen::MatrixXd Schur_A; // 柯西分解中的 S 矩阵，大小为：相机参数数 * 相机参数数
  Eigen::VectorXd Schur_B; // 柯西分解中的 r 向量，大小为：相机参数数 = 相机数 * N1，在 schur_complement 中该变量保存得到的相机增量 deltaPc
  bool update_parameter(double *step);
  bool checkParameter_camera(double *parameter_camera);
  bool checkParameter_point(double *parameter_point);
  void addParameterBlock(
      double *parameter_camera, double *parameter_point,
      BoundleAdjustment::CostFunction<N, N1, N2> *costfunction);
  void pre_process();
  void init_scaling();
  void solve();
  inline void schur_complement();
  // The function post_process used to copy the optimal pamameter to the double
  // array the user given.
  void post_process();
  static bool cmp(Residual_block *A, Residual_block *B);
  vector<vector<Residual_block *>> parameter_point_link; // 点参数与残差块之间的关联，按照点参数排序，每一项是与该点相关的残差块列表
  void removeParam(int param_id);  // this function will add in next version
  void out_schur_elimilate();      // this function will add in next version
  void incremental_optimal();      // this function will add in next version
};

template <int N, int N1, int N2>
Problem<N, N1, N2>::Problem() : parameter_a_size(0)
{
}
template <int N, int N1, int N2>
Problem<N, N1, N2>::~Problem()
{
  // delete the point we create
  for (int i = 0; i < parameter_camera_vector.size(); ++i)
  {
    delete (parameter_camera_vector[i]);
  }
  for (int i = 0; i < parameter_point_vector.size(); ++i)
  {
    delete (parameter_point_vector[i]);
  }
  for (int i = 0; i < residual_block_vector.size(); ++i)
  {
    delete (residual_block_vector[i]);
  }
}
template <int N, int N1, int N2>
bool Problem<N, N1, N2>::cmp(Residual_block *A, Residual_block *B)
{
  // Order the residual_block_vector point by the camera id
  if (A->camera_index < B->camera_index) return true;
  return false;
}

// 
template <int N, int N1, int N2>
void Problem<N, N1, N2>::pre_process()
{
  /* pre_process need to be done before schur_complement
   * in the process,the numbers of camera_index and point_index are known.Since
   * schur_complement cost much time we need to pre_process the struct of point
   * to camera. Notice that if the camera and point are not changed,the map
   * would only construct once. use the parameter_point_link,which is a link_list as
   * Point--->residual_node--->next residual_node
   *
   *                                                    |
   *                                                    |
   *                                                    V
   *                                                nextPoint
   * In each point we can find the pair<int Ci,int Cj>. Since the schur matrix
   * is Symmetric Matrix,we only need Ci<Cj so in this function we order the
   * link_list by camera id. Compute with the link happened in the function
   * schur_complement().
   */
  Schur_A.resize(parameter_a_size, parameter_a_size); // 初始化 Schur_A　为方阵
  Schur_B.resize(parameter_a_size); // 初始化 Schur_B 为列向量
  Schur_A.setZero();
  Schur_B.setZero();

  // 根据相机索引值由小到大重排
  for (int i = 0; i < parameter_point_link.size(); ++i)
  {
    sort(parameter_point_link[i].begin(), parameter_point_link[i].end(), cmp);
  }
}

// 
template <int N, int N1, int N2>
bool Problem<N, N1, N2>::update_parameter(double *step)
{
  // from cadidte to params and compute the step norm
  double step_norm = 0.0;
  for (int i = 0; i < parameter_camera_vector.size(); ++i)
  {
    parameter_camera_vector[i]->params = parameter_camera_vector[i]->candidate;
    parameter_camera_vector[i]->hessian.setZero();
    parameter_camera_vector[i]->residual.setZero(); // TODO: 该变量名需要修改，应为增量 g=JT*e
    step_norm = step_norm + parameter_camera_vector[i]->delta.norm();
  }
  for (int i = 0; i < parameter_point_vector.size(); ++i)
  {
    parameter_point_vector[i]->params = parameter_point_vector[i]->candidate;
    parameter_point_vector[i]->hessian.setZero();
    parameter_point_vector[i]->residual.setZero(); // TODO: 该变量名需要修改，应为增量 g=JT*e
    step_norm = step_norm + parameter_point_vector[i]->delta.norm();
  }
  (*step) = step_norm;
  if (step_norm < PARAMETERMIN)
  {
    return false;
  }
  return true;
}
template <int N, int N1, int N2>
bool Problem<N, N1, N2>::checkParameter_camera(double *parameter_camera)
{
  // we check if the double array had in the map<double,camera>
  if (parameter_camera_map.find(parameter_camera) != parameter_camera_map.end()) return true;
  return false;
}
template <int N, int N1, int N2>
bool Problem<N, N1, N2>::checkParameter_point(double *parameter_point)
{
  // we check if the double array had in the map<double,point>
  if (parameter_point_map.find(parameter_point) != parameter_point_map.end()) return true;
  return false;
}

/****************************************addParameterBlock********************************/
// 
template <int N, int N1, int N2>
void Problem<N, N1, N2>::addParameterBlock(
    double *parameter_camera, // 相机参数
    double *parameter_point, // 点参数
    BoundleAdjustment::CostFunction<N, N1, N2> *new_residual_node) // the costfunction
{
  // for saving time,we don't check the camera and point if there is
  // already a same tuple.So the input tuple must be different.

  if (!checkParameter_camera(parameter_camera))
  {
    Parameter<N1> *new_parameter = new Parameter<N1>();
    new_parameter->params =
        Eigen::Map<Eigen::Matrix<double, N1, 1>>(parameter_camera, N1);

    // 插入参数及其对应的索引
    parameter_camera_map.insert(
        std::pair<double *, int>(parameter_camera, parameter_camera_vector.size()));

    // 将参数插入到参数向量中
    parameter_camera_vector.push_back(new_parameter);

    // 更新参数向量的大小
    parameter_a_size = parameter_a_size + N1;
  }

  if (!checkParameter_point(parameter_point))
  {
    Parameter<N2> *new_parameter = new Parameter<N2>();
    new_parameter->params =
        Eigen::Map<Eigen::Matrix<double, N2, 1>>(parameter_point, N2);

    // 插入参数及其对应的索引
    parameter_point_map.insert(
        std::pair<double *, int>(parameter_point, parameter_point_vector.size()));

    // 将参数插入到参数向量中
    parameter_point_vector.push_back(new_parameter);

    // 
    vector<Residual_block *> parameter_point_list;
    parameter_point_link.push_back(parameter_point_list);
  }

  // 构建相机参数索引、点参数索引和代价函数的残差块之间的关联
  Residual_block *block =
      new Residual_block(parameter_camera_map[parameter_camera], // 相机参数索引
                          parameter_point_map[parameter_point],  // 点参数索引
                          new_residual_node);                    // 代价函数
  int id = block->point_index; // 点参数的索引
  parameter_point_link[id].push_back(block); // 将残差块插入到点参数对应的链表中
  residual_block_vector.push_back(block); // 将残差块插入到残差块向量中
}

/***************schur complement*************
 * 在该函数中完成 S 矩阵和 r 向量的计算
 * 通过柯西分解求解得到相机参数的增量 deltaPc
 */
template <int N, int N1, int N2>
void Problem<N, N1, N2>::schur_complement()
{
  /* schur_complement and the jacobi compute process is two time-cost processes.
   * We use the point_link to find the pair of camera which share one same
   * point. Notice Sij = sum(Wi^T*P^-1*Wj),so if we search pair by point ,the
   * Wi^T*P^-1 can keep in the second loop. As the schur matrix is symmetric.We
   * only compute the upper of S. By the way, the Schur_B can be computed by the
   * same process. finally because the Schur_A is a positive symmetric matrix,we
   * can use the llt() to solve delta camera.
   */
  int length_camera = parameter_camera_vector.size();           // 相机参数数量
  int parameter_point_link_size = parameter_point_link.size();  // 3D点参数数量

  for (int i = 0; i < parameter_point_link_size; ++i)
  {
    int inner_size = parameter_point_link[i].size();  // 该3D点对应的残差块数量
    for (int j = 0; j < inner_size; ++j)
    {
      int id_1 = parameter_point_link[i][j]->camera_index;

      // 计算
      Eigen::Matrix<double, N1, N2> WT_Einv; 

      // 计算 W*U^(-1)
      WT_Einv.noalias() = parameter_point_link[i][j]->hessian_W.lazyProduct(
          parameter_point_vector[i]->hessian_inverse); // 矩阵 U 的逆矩阵

      // 计算向量 r 完整的第二项：W*U^(-1)*Jp*e
      Schur_B.segment<N1>(id_1 * N1).noalias() -=
          WT_Einv.lazyProduct(parameter_point_vector[i]->residual); // TODO: 该变量名需要修改，应为增量 g=JT*e

      // 计算 Schur 矩阵的第二项：W*U^(-1)*W^T
      for (int k = j; k < inner_size; ++k)
      {
        int id_2 = parameter_point_link[i][k]->camera_index;
        Schur_A.block<N1, N1>(id_1 * N1, id_2 * N1).noalias() -=
            WT_Einv.lazyProduct(parameter_point_link[i][k]->hessian_W.transpose());
      }
    }
  }

  // 求解 S*deltaPc = r，即 Schur_A*deltaPc = Schur_B
  // 将求解结果 deltaPc 保存在 Schur_B 中（两者size相同）
  Schur_B = Schur_A.selfadjointView<Eigen::Upper>().llt().solve(Schur_B);
  for (int i = 0; i < length_camera; ++i)
  {
    parameter_camera_vector[i]->delta = Schur_B.segment<N1>(i * N1);
  }
}

/****************************************end schur
 * complement********************************/
template <int N, int N1, int N2>
void Problem<N, N1, N2>::init_scaling()
{
  // jacobi_scaling = 1/(1+sqrt(jacobi_scaling )) plus the one to avoid / zero
  int parameter_camera_length = parameter_camera_vector.size();
  int parameter_point_length = parameter_point_vector.size();
  for (int i = 0; i < parameter_camera_length; ++i)
  {
    parameter_camera_vector[i]->jacobi_scaling =
        1.0 / (1.0 + sqrt(parameter_camera_vector[i]->jacobi_scaling.array()));
  }
  for (int i = 0; i < parameter_point_length; ++i)
  {
    parameter_point_vector[i]->jacobi_scaling =
        1.0 / (1.0 + sqrt(parameter_point_vector[i]->jacobi_scaling.array()));
  }
}

template <int N, int N1, int N2>
void Problem<N, N1, N2>::solve()
{
  int outcount = 0;
  double minimum_cost = -1;
  double model_cost = 0.0;
  int num_consecutive_nonmonotonic_steps = 0;
  double accumulated_reference_model_cost_change = 0;
  double accumulated_candidate_model_cost_change = 0;
  double current_cost;
  double candidate_cost;
  double reference_cost;
  double Miu = INITMIU;
  double v0 = 2.0;
  clock_t t1 = clock();
  int parameter_camera_vector_length = parameter_camera_vector.size(); // 相机参数数量
  int parameter_point_vector_length = parameter_point_vector.size(); // 点参数数量
  int residual_node_length = residual_block_vector.size(); // 残差块数量
  double residual_cost = 0.0; // 总的代价，即所有残差的平方和

  // 输出相机参数和点参数的数量，以及残差块的数量
  cout << "parameter_camera_vector_length: " << parameter_camera_vector_length << endl;
  cout << "parameter_point_vector_length: " << parameter_point_vector_length << endl;
  cout << "residual_node_length: " << residual_node_length << endl;

  // 对于每一个残差块（观测点）
  for (int i = 0; i < residual_node_length; ++i)
  {
    // 计算残差块的雅可比矩阵和残差
    residual_block_vector[i]->residual_node->computeJacobiandResidual(
        &parameter_camera_vector[residual_block_vector[i]->camera_index]->params, // 相机参数，内参外参
        &parameter_point_vector[residual_block_vector[i]->point_index]->params, // 3D点参数
        &residual_block_vector[i]->jacobi_parameter_camera, // 相机内外参数的雅可比矩阵
        &residual_block_vector[i]->jacobi_parameter_point,  // 3D点参数的雅可比矩阵
        &residual_block_vector[i]->residual); // 残差，重投影误差

    // 计算总的代价，即所有残差的平方和
    residual_cost = residual_cost + residual_block_vector[i]->residual.squaredNorm(); //  L2 范数的平方,即平方和

    // 计算相机参数和点参数的雅可比矩阵的缩放因子
    parameter_camera_vector[residual_block_vector[i]->camera_index]->jacobi_scaling +=
        residual_block_vector[i]->jacobi_parameter_camera.colwise().squaredNorm(); // 求每一列的平方和
    parameter_point_vector[residual_block_vector[i]->point_index]->jacobi_scaling +=
        residual_block_vector[i]->jacobi_parameter_point.colwise().squaredNorm();
  }
  residual_cost /= 2;
  pre_process();
  init_scaling();
  cout << "iteration|    new_residual    |    old_residual    |    step_norm   "
          " |    radius    |    iter time    "
       << endl;
  while (outcount < OUTTER_ITERATION_NUMBER)
  {
    for (typename std::vector<
             BoundleAdjustment::Problem<N, N1, N2>::Residual_block *>::iterator
             it = residual_block_vector.begin();
         it != residual_block_vector.end(); ++it)
    {
      // 
      (*it)->jacobi_parameter_camera = (*it)->jacobi_parameter_camera.array().rowwise() *
                                  parameter_camera_vector[(*it)->camera_index]
                                      ->jacobi_scaling.transpose()
                                      .array();
                                      
      // 
      (*it)->jacobi_parameter_point = (*it)->jacobi_parameter_point.array().rowwise() *
                                  parameter_point_vector[(*it)->point_index]
                                      ->jacobi_scaling.transpose()
                                      .array();

      // 计算 W 矩阵 = JcT * Jp
      (*it)->hessian_W.noalias() =
          (*it)->jacobi_parameter_camera.transpose().lazyProduct(
              (*it)->jacobi_parameter_point);
    }
    ++outcount;
    double totaltime = (double)(clock() - t1) / CLOCKS_PER_SEC;
    t1 = clock();
    if (outcount <= 1)
    {
      candidate_cost = residual_cost;
      reference_cost = residual_cost;
    }
    current_cost = residual_cost;
    accumulated_candidate_model_cost_change += model_cost;
    accumulated_reference_model_cost_change += model_cost;
    if (outcount == 1 || current_cost < minimum_cost)
    {
      minimum_cost = current_cost;
      num_consecutive_nonmonotonic_steps = 0;
      candidate_cost = current_cost;
      accumulated_candidate_model_cost_change = 0;
    }
    else
    {
      ++num_consecutive_nonmonotonic_steps;
      if (current_cost > candidate_cost)
      {
        candidate_cost = current_cost;
        accumulated_candidate_model_cost_change = 0.0;
      }
    }
    if (num_consecutive_nonmonotonic_steps ==
        max_consecutive_nonmonotonic_steps)
    {
      reference_cost = candidate_cost;
      accumulated_reference_model_cost_change =
          accumulated_candidate_model_cost_change;
    }
    /*****************begin the inner loop******************************/
    int innercount = 0;
    double r = -1.0; // 阻尼系数 rho

    double new_residual_cost;
    while (innercount < INNER_ITERATION_NUMBER)
    {
      ++innercount;
      // do make schur_matrix residual_c parameter_2_hessian init
      Schur_A.setZero();
      Schur_B.setZero();
      for (typename std::vector<BoundleAdjustment::Problem<
               N, N1, N2>::Residual_block *>::iterator it =
               residual_block_vector.begin();
           it != residual_block_vector.end(); ++it)
      {
        int id_camera = (*it)->camera_index;
        int id_point = (*it)->point_index;

        // 矩阵 S 的第一项 V 矩阵的每一个对角块 = JcT * Jc
        Schur_A.block<N1, N1>(id_camera * N1, id_camera * N1).noalias() +=
            (*it)->jacobi_parameter_camera.transpose().lazyProduct(
                (*it)->jacobi_parameter_camera);

        // 向量r的第一项：JcT * e (e 为残差）
        Schur_B.segment<N1>(id_camera * N1).noalias() -=
            (*it)->jacobi_parameter_camera.transpose().lazyProduct((*it)->residual);

        // 矩阵U = JpT * Jp
        parameter_point_vector[id_point]->hessian.noalias() +=
            (*it)->jacobi_parameter_point.transpose().lazyProduct(
                (*it)->jacobi_parameter_point);

        // 向量r的第二项的后部：JpT * e，完整的第二项在 schur_complement() 中计算
        parameter_point_vector[id_point]->residual.noalias() -=
            (*it)->jacobi_parameter_point.transpose().lazyProduct((*it)->residual);
      } // 内循环结束

      // 更新 S 的第一项 V 矩阵的对角块，即加上 DTD 矩阵
      Schur_A.diagonal().noalias() += 1 / Miu * Schur_A.diagonal();

      // 更新矩阵 U 的对角块，即加上 DTD 矩阵
      // 计算矩阵 U 的逆矩阵
      for (int i = 0; i < parameter_point_vector_length; ++i)
      {
        // update the parameter_2_hessian by Miu and compute the inverse
        parameter_point_vector[i]->hessian.diagonal().noalias() +=
            1 / Miu * parameter_point_vector[i]->hessian.diagonal();
        parameter_point_vector[i]->hessian_inverse =
            parameter_point_vector[i]->hessian.inverse();
      }

      schur_complement();

      // 计算 deltaPp = U^(-1) * (Jp*e - W^T * deltaPc)
      for (int i = 0; i < residual_node_length; ++i)
      {
        int id = residual_block_vector[i]->point_index;

        // 计算 Jp*e - W^T * deltaPc
        parameter_point_vector[id]->residual.noalias() -=
            residual_block_vector[i]->hessian_W.transpose().lazyProduct(
                parameter_camera_vector[residual_block_vector[i]->camera_index]->delta);
      }

      // 计算 deltaPp = U^(-1) * (Jp*e - W^T * deltaPc)
      for (int i = 0; i < parameter_point_vector_length; ++i)
      {
        parameter_point_vector[i]->delta.noalias() =
            parameter_point_vector[i]->hessian_inverse.lazyProduct(
                parameter_point_vector[i]->residual);
      }
      model_cost = 0.0;

      for (typename std::vector<BoundleAdjustment::Problem<
               N, N1, N2>::Residual_block *>::iterator it =
               residual_block_vector.begin();
           it != residual_block_vector.end(); ++it)
      {
        Eigen::Matrix<double, N, 1> delta_parameter;
        int id_1 = (*it)->camera_index;
        int id_2 = (*it)->point_index;
        delta_parameter = (*it)->jacobi_parameter_camera.lazyProduct(
                              parameter_camera_vector[id_1]->delta) +
                          (*it)->jacobi_parameter_point.lazyProduct(
                              parameter_point_vector[id_2]->delta);

        // 阻尼系数 rho 的分母，这里计算方式与论文中不同
        model_cost += (delta_parameter.transpose() *
                       (2 * (*it)->residual + delta_parameter))(0, 0);
      }
      model_cost = model_cost / 2;
      cout << "model cost:" << model_cost << endl;
      if (model_cost < 0)
      {
        for (int i = 0; i < parameter_camera_vector_length; ++i)
        {
          parameter_camera_vector[i]->delta.array() *=
              parameter_camera_vector[i]->jacobi_scaling.array();
        }
        for (int i = 0; i < parameter_point_vector_length; ++i)
        {
          parameter_point_vector[i]->delta.array() *=
              parameter_point_vector[i]->jacobi_scaling.array();
        }
      }
      new_residual_cost = 0.0;
      for (typename std::vector<BoundleAdjustment::Problem<
               N, N1, N2>::Residual_block *>::iterator it =
               residual_block_vector.begin();
           it != residual_block_vector.end(); ++it)
      {
        int id_1 = (*it)->camera_index;
        int id_2 = (*it)->point_index;

        // 计算相机参数的候选值和点参数的候选值
        parameter_camera_vector[id_1]->candidate =
            parameter_camera_vector[id_1]->params + parameter_camera_vector[id_1]->delta;
        parameter_point_vector[id_2]->candidate =
            parameter_point_vector[id_2]->params + parameter_point_vector[id_2]->delta;

        // 使用机参数的候选值和点参数的候选值计算新的残差
        (*it)->residual_node->computeResidual(
            &parameter_camera_vector[id_1]->candidate,
            &parameter_point_vector[id_2]->candidate, &new_residual_cost);
      }
      new_residual_cost /= 2;

      // 阻尼系数 rho ,这里计算方式与论文中差了个负号
      double relative_decrease =
          (new_residual_cost - current_cost) / model_cost;
      double historical_relative_decrease =
          (new_residual_cost - reference_cost) /
          (accumulated_reference_model_cost_change + model_cost);
      r = relative_decrease < historical_relative_decrease
              ? historical_relative_decrease
              : relative_decrease;

      if (r > 0.001)
      {
        Miu = min(Miu / max(1.0 / 3.0, 1.0 - pow((2 * r - 1.0), 3)), 1e16);
        v0 = 2;
        break;
      }
      else
      {
        Miu = Miu / v0;
        v0 *= 2;
      }
    }
    /*****************end the inner loop********************************/
    if ((residual_cost - new_residual_cost) / residual_cost < MINDIF)
    {
      cout << "leave by MINDIF reached!" << endl;
      break;
    }
    double step = 0.0;
    if (!update_parameter(&step))
    {
      cout << "reach the parameter limit :" << PARAMETERMIN << endl;
      break;
    }
    for (typename std::vector<
             BoundleAdjustment::Problem<N, N1, N2>::Residual_block *>::iterator
             it = residual_block_vector.begin();
         it != residual_block_vector.end(); ++it)
    {
      (*it)->residual_node->computeJacobiandResidual(
          &parameter_camera_vector[(*it)->camera_index]->params,
          &parameter_point_vector[(*it)->point_index]->params,
          &(*it)->jacobi_parameter_camera, &(*it)->jacobi_parameter_point,
          &(*it)->residual);
    }
    printf("%4s(%d)", std::to_string(outcount).c_str(), innercount);
    printf("%20s", std::to_string(new_residual_cost).c_str());
    printf("%20s", std::to_string(residual_cost).c_str());
    printf("%20s", std::to_string(step).c_str());
    printf("%20s", std::to_string(r).c_str());
    printf("%20s", std::to_string(totaltime).c_str());
    printf("\n");
    residual_cost = new_residual_cost;
  }
  post_process();
}

/*
  map<double *, int> parameter_camera_map;
  iter->first : double *parameter_camera
  iter->second : int index
*/
template <int N, int N1, int N2>
void Problem<N, N1, N2>::post_process()
{
  map<double *, int>::iterator iter;
  for (iter = parameter_camera_map.begin(); iter != parameter_camera_map.end(); ++iter)
  {
    Eigen::Map<Eigen::Matrix<double, N1, 1>>(
        iter->first, parameter_camera_vector[iter->second]->params.rows(), 1) =
        parameter_camera_vector[iter->second]->params;
  }
  for (iter = parameter_point_map.begin(); iter != parameter_point_map.end(); ++iter)
  {
    Eigen::Map<Eigen::Matrix<double, N2, 1>>(
        iter->first, parameter_point_vector[iter->second]->params.rows(), 1) =
        parameter_point_vector[iter->second]->params;
  }
}
}  // namespace BoundleAdjustment
