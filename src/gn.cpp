#include <iostream>
#include <random>
#include <iterator>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
// #include <opencv2/core/core.hpp>
#include <Eigen/QR>

#define Sample_NUM 100

using namespace std;
using namespace Eigen;

// 估计函数  y(x) = exp( a*x*x + b*x + c ) 
int main() {
	// cv::RNG rng; 
	std::vector<double> x_set, y_set;
	double a = 1, b =2 , c = 1;			// 待估计参数

	srand(time(NULL));
	double seed = (rand() % (10-1)) + 1; 
	for (int i = 1; i<= Sample_NUM; ++i) {

		// x样本
		double x = (double)i/100.0;
		x_set.push_back(x); 

		// Define random generator with Gaussian distribution
		double mean = 0.0;		//均值
		// double stddev = (double)seed/10;	//标准差0-1
		double stddev = 0.10;//标准差
		std::default_random_engine generator; 
		std::normal_distribution<double> gauss(mean, stddev);

		// y样本   // Add Gaussian noise
		double y =  exp( a*x*x + b*x + c )  + gauss(generator);

		y_set.push_back(y);

		// opencv随机数产生，每次随机都一样，真随机？？？？？
		// double x = (double)i/100 ;
     	// x_set.push_back( x );
        // y_set.push_back( exp( a*x*x + b*x + c ) + rng.gaussian(1.0)  ); 
	}
	
    // // Output the result, for demonstration purposes
	// std::copy(begin(x_set), end(x_set), std::ostream_iterator<double>(std::cout, " "));
    // std::cout << "\n";
    // std::copy(begin(y_set), end(y_set), std::ostream_iterator<double>(std::cout, " "));
    // std::cout << "\n";

	// gn 算法
	static double aft_a = 0, aft_b = 0, aft_c = 0;
	Eigen::Vector3d delta_abc;
	Eigen::MatrixXd Jacb_abc, error_i;
	Eigen::Matrix3d H; 
	Eigen::Vector3d g;
	double curCost;

	if (x_set.size() != Sample_NUM || y_set.size() != Sample_NUM) {

		std::cerr << "data error!\n";
		return -1;
	}

	int iteratorNum = 100;	// 迭代次数
	for(int i = 1; i<=iteratorNum; i++) {
		for (int j = 0; j< Sample_NUM; ++j) {
			Jacb_abc.resize(Sample_NUM, 3);
			error_i.resize(Sample_NUM, 1);

			double y_est = exp( aft_a * x_set.at(j) * x_set.at(j) + 	\
													aft_b * x_set.at(j) + aft_c);
			// 雅克比  对待优化变量的偏导 注意真实的J应该是
			Jacb_abc(j,0) =   -x_set.at(j) * x_set.at(j) * y_est;
			Jacb_abc(j,1) =   			    -x_set.at(j) * y_est;
			Jacb_abc(j,2) = 					    -1.0 * y_est;
		
			// 误差
			error_i(j, 0) = y_set.at(j) - y_est;
		}	
		// 计算增量方程 H * delta_? = g	
		H = Jacb_abc.transpose() * Jacb_abc;		// 计算H
		g = -Jacb_abc.transpose() * error_i; 		// 计算g 
		delta_abc = H.ldlt().solve(g); 
		
		// 误差all
		curCost = 1.0/2 * (error_i.transpose() * error_i)(0,0);

		if (i%1==0) {

			std::cout << "当前迭代次数: " << i << "/" << iteratorNum << "     当前总误差: " << curCost  << std::endl;
			std::cout << "   当前增量: " << delta_abc.transpose() << std::endl;

		}

		if ( isnan(delta_abc(0)) || isnan(delta_abc(1)) || isnan(delta_abc(2)))
			break; 
	

		// 判断是否提前收敛 增量是否足够小
		if ((delta_abc.lpNorm<1>() < 1e-6*(fabs(aft_a)+fabs(aft_b)+fabs(aft_c) + 1e-6))) {
			break;
		}

		// 更新增量
		aft_a += delta_abc[0];
		aft_b += delta_abc[1];
		aft_c += delta_abc[2];

	}

	std::cout << "真实abc: " << a << " " << b << " " << c << std::endl;
	std::cout << "迭代结束! \n" << "a: " << aft_a << "\nb: " << aft_b 
							<< "\nc: " << aft_c << std::endl;

	return 0;
}
