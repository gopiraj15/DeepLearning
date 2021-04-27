#include "DeepNeuralNetwork.h"

using namespace cv;
using namespace std;



//!< This function initializes the parameters for the deep neural network
void DeepNeuralNetwork::initialize_deep_parameters(cv::Mat X, const std::vector<int>& layer_dims_)
{
	m = X.cols;

	layer_dims = layer_dims_;

	int L = layer_dims.size();

	for (int i = 1; i < L; ++i)
	{
		weights.emplace_back(cv::Mat::ones(layer_dims[i], layer_dims[i - 1], CV_64FC1));
		biases.emplace_back(cv::Mat::zeros(layer_dims[i], 1, CV_64FC1));
	}

	for (int i = 0; i < weights.size(); ++i)
	{
		randn(weights[i], cv::Scalar(0), cv::Scalar(0.01 * m / 2));
	}


	Z.resize(weights.size());
	A.resize(weights.size());

	weight_grads.resize(weights.size());
	bias_grads.resize(weights.size());

}

void DeepNeuralNetwork::linear_forward(cv::Mat& A, cv::Mat& W, cv::Mat& b, cv::Mat& Z)
{
	m = A.cols;
	//broadcasting
	cv::Mat b1;
	std::vector<cv::Mat> tmpB;
	for (int i = 0; i < m; ++i)
	{
		tmpB.emplace_back(b);
	}
	cv::merge(tmpB, b1); tmpB.clear();

	Z = W * A + b1.reshape(1, W.rows);
}

void DeepNeuralNetwork::linear_activation_forward(cv::Mat A_prev, cv::Mat W, cv::Mat b, cv::Mat& A, const std::string& activation)
{
	cv::Mat Z;
	linear_forward(A_prev, W, b, Z);
	if (activation == "sigmoid")
	{
		A = sigmoid(Z);
	}
	else if (activation == "relu")
	{
		A = relu(Z);
	}
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


//computes relu on the input
cv::Mat relu(cv::Mat ip)
{
	return cv::Mat(cv::max(ip, 0.0));
}


cv::Mat col_sum(cv::Mat ip, double m)
{
	cv::Mat res;
	cv::reduce(ip, res, 1, REDUCE_SUM);

	return res / m;
}
