#include "NeuralNetwork.h"

using namespace cv;
using namespace std;



void load_dataset(std::string x_data, std::string y_data, vector<double>& X, vector<double>& Y)
{
	ifstream xf(x_data), yf(y_data);

	while (!xf.eof())
	{
		double x;
		xf >> x;
		X.emplace_back(x);
	}

	xf.close();

	X.erase(X.begin() + X.size() - 1);

	while (!yf.eof())
	{
		double y;
		yf >> y;
		Y.emplace_back(y);
	}

	yf.close();

	Y.erase(Y.begin() + Y.size() - 1);

}

//This function computes tanh on the input
cv::Mat tanh(cv::Mat ip)
{
	Mat p_exp, n_exp;
	cv::exp(ip, p_exp);
	cv::exp(-ip, n_exp);
	return (p_exp - n_exp) / (p_exp + n_exp);
}

//computes sigmoid on the input
cv::Mat sigmoid(cv::Mat ip)
{
	Mat n_exp;
	cv::exp(-ip, n_exp);
	return cv::Mat(1 / (1 + n_exp));
}

cv::Mat col_sum(cv::Mat ip, double m)
{
	cv::Mat res;
	cv::reduce(ip, res, 1, REDUCE_SUM);

	return res / m;
}

