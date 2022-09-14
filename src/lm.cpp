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
	double a = 1, b = 2 , c = 3;			// 待估计参数

	srand(time(NULL));
	double seed = (rand() % (10-1)) + 1; 
	for (int i = 1; i<= Sample_NUM; ++i) {

		// x样本
		double x = (double)i/100.0;
		x_set.push_back(x); 

		// Define random generator with Gaussian distribution
		double mean = 0.0;		//均值
		// double stddev = (double)seed/10;	//标准差0-1
		double stddev = 1.0;//标准差
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

	// lm 算法
	static double aft_a = 0.0, aft_b = 0.0, aft_c = 0.0;
	Eigen::Vector3d delta_abc;
	Eigen::MatrixXd Jacb_abc, error_i, error_sum_aft;
	Eigen::Matrix3d H; 
	Eigen::Vector3d g;
	double curSquareCost, curSquareCost_aft;
	double taylar_similary = 0;			// 一阶泰勒近似度
	Eigen::Matrix3d D = Eigen::Matrix3d::Identity();		// 与H同维度	
	double u = 1.0, v = 2;		// 拉格朗日乘子，v控制u大小 

	if (x_set.size() != Sample_NUM || y_set.size() != Sample_NUM) {

		std::cerr << "data error!\n";
		return -1;
	}

	int iteratorNum = 100;	// 迭代次数
	for(int i = 1; i<=iteratorNum; i++) {

		for (int j = 0; j< Sample_NUM; ++j) {		// 构造方程
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
	
		// 误差平方和
		curSquareCost = 1.0f/2*(error_i.transpose() * error_i)(0,0);
		// std::cout << curSquareCost << std::endl;	

		// 计算初始的u
		// 初值比较好，t取小；初值未知，t取大
		// 去对角线最大元素作为初值
		static bool isFirstStart = true;
		if (isFirstStart) {
			
			double diag_max_element = 0.0;
			for (int m = 0; m<H.rows(); ++m ) {
				if (H(m,m) >= diag_max_element) {
					diag_max_element = H(m,m);
				}
			}
			double t = 1.0;		//1e-6  1e-3  或者 1.0
			u = t * diag_max_element;
			isFirstStart = false;
		}

		// 计算增量方程 （H + namDa*D_pow2） * delta_? = g
		Eigen::Matrix3d D_pow2 = Eigen::Matrix3d::Identity();
		delta_abc = (H + u*D_pow2).ldlt().solve(g); 

		if ( isnan(delta_abc(0)) || isnan(delta_abc(1)) || isnan(delta_abc(2)))
			continue; 
	
		// 判断是否提前收敛 
		if ( (delta_abc.lpNorm<1>() < 1e-6*(fabs(aft_a)+fabs(aft_b)+fabs(aft_c) + 1e-6)) || 
								(g.lpNorm<Eigen::Infinity>() < 1e-6) ) { 

			std::cout << "真实abc: " << a << " " << b << " " << c << std::endl;
			std::cout << "迭代结束! \n" << "a: " << aft_a << "\nb: " << aft_b 
							<< "\nc: " << aft_c << std::endl;
			return 0;
		}

		// 求一阶泰勒近似度
		error_sum_aft.resize(Sample_NUM, 1);
		for (int k = 0; k<Sample_NUM; k++) {
			error_sum_aft(k,0) = y_set.at(k) - exp( (aft_a+delta_abc(0)) * x_set.at(k) * x_set.at(k) + 	\
													(aft_b+delta_abc(1)) * x_set.at(k) + (aft_c+delta_abc(2)) );
		}	
		curSquareCost_aft = 1.0/2 * error_sum_aft.squaredNorm();

		// 以下三种计算近似度都可以

		Eigen::MatrixXd L0_Ldelta = -delta_abc.transpose() * Jacb_abc.transpose() * error_i - 1.0f/2 * delta_abc.transpose() * Jacb_abc.transpose() * Jacb_abc * delta_abc;
		// Eigen::MatrixXd L0_Ldelta = 1.0f/2 * delta_abc.transpose() * ( u*delta_abc + g);  
		// Eigen::MatrixXd L0_Ldelta = 1.0/2*((Jacb_abc*delta_abc).transpose() * (Jacb_abc*delta_abc));
		
		taylar_similary = (curSquareCost-curSquareCost_aft) / L0_Ldelta(0,0); 
		if (taylar_similary>0) { 	// 近似可用 
			// 更新增量
			aft_a += delta_abc(0);
			aft_b += delta_abc(1);
			aft_c += delta_abc(2);

			// 更新u
			u = u * std::max<double>(1.0/3, 1-pow(2*taylar_similary-1, 3));
			v = 2.0;
			
		}else{
			// 近似不可用  采用一阶梯度
			u = v*u;
			v = 2*v;		
		}


		if (i%1==0) 
		{
			std::cout << "   泰勒近似: " << taylar_similary << std::endl;
			std::cout << "   本次增量: " << delta_abc.transpose() << std::endl;
			std::cout << "当前迭代次数: " << i << "/" << iteratorNum << "     当前总误差: " << curSquareCost << std::endl<< std::endl;
		}

	}

	std::cout << "真实abc: " << a << " " << b << " " << c << std::endl;
	std::cout << "迭代结束! \n" << "a: " << aft_a << "\nb: " << aft_b 
							<< "\nc: " << aft_c << std::endl;

	return 0;
}
