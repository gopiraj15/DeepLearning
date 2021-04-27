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

//This function implements a L-model neural network
void DeepNeuralNetwork::L_model_forward(cv::Mat X, std::vector<cv::Mat>& weights, std::vector<cv::Mat>& biases, cv::Mat& AL)
{

	Mat A = X;
	int L = weights.size();

	// Implement [LINEAR -> RELU]*(L-1).
	for (int l = 0; l < L - 1; ++l)
	{
		Mat A_prev = A.clone();
		linear_activation_forward(A_prev, weights[l], biases[l], A, "relu");
	}


	// Implement LINEAR -> SIGMOID.
	linear_activation_forward(A, weights[L-1], biases[L-1], AL, "sigmoid");

}

double DeepNeuralNetwork::compute_cost(cv::Mat AL, cv::Mat Y)
{

	double m_ = Y.cols;

	// cost = -np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))/m 
	cv::Mat p_A2, p_m_A2;
	cv::log(AL, p_A2); cv::log(1 - AL, p_m_A2);

	cv::Mat logprobs = p_A2.mul(Y) + p_m_A2.mul(1 - Y);
	return cv::sum(logprobs)[0] / (-m_);
}


//! performs linear backpropagation
void DeepNeuralNetwork::linear_backward(cv::Mat dZ, cv::Mat A_prev, cv::Mat W, cv::Mat b,
	cv::Mat& dA_prev, cv::Mat& dW, cv::Mat& db)
{
	double m_ = A_prev.cols;

	dW = (dZ * A_prev.t()) / m_;
	db = col_sum(dZ, m_);
	dA_prev = W.t() * dZ;
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
