#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>   //time
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>//一元边
#include <g2o/core/base_binary_edge.h>//二元边
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>  //求解方法
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>

using namespace std;

/*
_vertices[]：存储顶点编号
_estimate：存储待估计的优化变量
_measurement：观测值，就是数据点的y坐标
_error：_measurement - 预测值 
*/

// 顶点，即优化变量，模板参数：优化变量维度和数据类型
// 必须继承于此类g2o::BaseVertex，描述待估计变量
// Vertex：顶点， CurveFitting:曲线拟合 待估计参数3维
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {

public:    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW         ////这个宏使“new Foo”总是返回一个对齐的指针

//继承需要重写的虚函数    
    virtual void setToOriginImpl()      //顶点重置函数，设定被优化变量的原始值 abc[3]
    {
        _estimate << 0,0,0;
    }
    
    virtual void oplusImpl( const double* update )  //表达顶点的更新（待优化变量的更新）主要用于优化过程中增量△x 的计算。我们根据增量方程计算出增量之后，就是通过这个函数对估计值进行调整的
    {
        _estimate += Eigen::Vector3d( update );
    }

    // 存盘和读盘：留空
    virtual bool read( std::istream& in ) {}
    virtual bool write( std::ostream& out ) const {}


};
// 曲线模型的边，即误差模型 模板参数：观测值维度，类型，连接顶点类型
// 必须继承于此类g2o::BaseUnaryEdge
// 一元边 BaseUnaryEdge< 1, double, CurveFittingVertex >
// 二元边 BaseBinaryEdge<2,double,CurveFittingVertex>
// 多元边 BaseMultiEdge<>
class CurveFittingEdge: public g2o::BaseUnaryEdge< 1, double, CurveFittingVertex >
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge( double x ) :  BaseUnaryEdge(), _x(x) {}
    
    virtual void computeError()     // 计算曲线模型误差
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex *> (_vertices[0]);   //_vertices存储的顶点编号
        const Eigen::Vector3d abc = v->estimate();      //获取顶点0的优化变量
        
        double now_val = std::exp( abc(0)*_x*_x + abc(1)*_x + abc(2) );     //当前优化变量下的曲线y值
        _error(0) = _measurement - now_val;   //y值为_measurement

     //   _error(0,0) = _measurement - std::exp( abc(0,0)*_x*_x + abc(1,0)*_x + abc(2,0) );
        
    }

    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}

public:
    double _x;  // x值 
};

int main(int argc, char** argv) {

    int N = 100;
    double abc[3] = {0,0,0};    //待估计参数

    double a = 1.0, b = 2.0, c = 1.0;   //曲线实际参数
    double w_sigma = 0.1;       //噪声
    cv::RNG rng;             //opencv随机数产生器
//样本数据产生
    vector<double> x_data, y_data;
    cout << "产生的样本数据如下：" << endl;

    for (int i = 0; i< N; ++i) {
        
        double x = (double)i/100.0;

        x_data.push_back( x );
        y_data.push_back( exp( a*x*x + b*x + c ) + rng.gaussian(w_sigma)  );    //cmath库 
        
        cout<<x_data[i]<<" "<<y_data[i]<<endl;
    }
//图优化
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<3, 1> >  Block;    // 每个误差项优化变量维度为3，误差值维度为1
/*
LinearSolverCholmod ：使用sparse cholesky分解法。继承自LinearSolverCCS
LinearSolverCSparse：使用CSparse法。继承自LinearSolverCCS
LinearSolverPCG ：使用preconditioned conjugate gradient 法，继承自LinearSolver
LinearSolverDense ：使用dense cholesky分解法。继承自LinearSolver
LinearSolverEigen： 依赖项只有eigen，使用eigen中sparse Cholesky 求解，因此编译好后可以方便的在其他地方使用，性能和CSparse差不多。继承自LinearSolver
*/
  /*  Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();  //矩阵求解方式
    //Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); //构建线性方程求解器

    Block* solver_ptr = new Block( linearSolver );      // 矩阵块求解器

    // 梯度下降方法: 从GN（高斯牛顿）, LM(列文夸克), DogLeg（信赖区域一种） , 选一种方式
    // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr ); 
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
    // g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );
*/

	std::unique_ptr<Block::LinearSolverType> linearSolver ( new g2o::LinearSolverDense<Block::PoseMatrixType>() );
	std::unique_ptr<Block> solver_ptr (new Block( std::move(linearSolver) ));
	//g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( std::move(solver_ptr) );
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( std::move(solver_ptr) );
	//g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( std::move(solver_ptr) );

// 真实值：1 2 1		
// dg: 0.890911   2.1719 0.943629
// lm: 0.89338  2.16831 0.944841
// gn: 71.3575 -29.4185  -4.0554
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm( solver );   // 设置求解器
    optimizer.setVerbose( true );       // 打开调试输出

    //设置顶点的初始值，并添加
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate( Eigen::Vector3d(0,0,0) );       //设定初始值（迭代初始值）
    v->setId(0);                    //设置顶点的id，等待后续边进行链接: edge->setVertex( 0, v );   
    optimizer.addVertex(v);     //添加顶点到优化器
    
    //往图中增加边，并添加
    for ( int i=0; i<N; i++ )
    {
        CurveFittingEdge* edge = new CurveFittingEdge( x_data[i] );         //初始化每条边，传入边值x
        edge->setId(i);                // 定义边的编号, 决定在H矩阵中的位置
        edge->setVertex( 0, v );                // 设置连接的顶点
        edge->setMeasurement( y_data[i] );      // 传入观测数值

        //传入误差的协方差矩阵 试过只传Eigen::Matrix<double,1,1>::Identity()也可以
        edge->setInformation( Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma) ); // 信息矩阵：协方差矩阵之逆
        //edge->setInformation( Eigen::Matrix<double,1,1>::Identity() );
        optimizer.addEdge( edge );  //添加边到优化器
    }
    
    // 执行优化
    cout << "start optimization: " << endl;
    //chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    optimizer.initializeOptimization();     //初始化以上配置
    optimizer.optimize(10);     //启动优化器 迭代次数限制在多少次以内

    //chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    //chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );  
    //cout<<"solve time cost = "<< time_used.count()<<" seconds. "<<endl;
    
    // 输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();       //g2o::BaseVertex< 3, Eigen::Vector3d >
    cout << "Estimated model: " << abc_estimate.transpose() << endl;
    
    return 0;

}


