#ifndef __DEEP_NEURAL_NETWORK_H__
#define __DEEP_NEURAL_NETWORK_H__
#pragma once

#include <opencv2/opencv.hpp>

//! Activation functions
//This function computes tanh on the input Mat
cv::Mat tanh(cv::Mat ip);


//computes sigmoid on the input Mat
cv::Mat sigmoid(cv::Mat ip);

//computes relu on the input Mat
cv::Mat relu(cv::Mat ip);

//computes the sum of the columns and scales the output with m
cv::Mat col_sum(cv::Mat ip, double m);



class DeepNeuralNetwork
{
public:
	std::vector<cv::Mat> weights; //!< Weights of different layers
	std::vector<cv::Mat> biases;//!< biases of different layers
	std::vector<int> layer_dims; //!< dimensions of different layers
	double m; //!< number of training samples
	cv::Mat predictions; //!< predicted outputs

	
	//!< This function initializes the parameters for the deep neural network
	void initialize_deep_parameters(cv::Mat X, const std::vector<int>& layer_dims);
	

	//!< linear forward propagation
	void linear_forward(cv::Mat &A, cv::Mat &W, cv::Mat &b, cv::Mat &Z);

	//!< linear forward activation
	void linear_activation_forward(cv::Mat A_prev, cv::Mat W, cv::Mat b, cv::Mat& A, const std::string& activation = "sigmoid");


	//!< implementing a L-layer neural network
	void L_model_forward(cv::Mat X, std::vector<cv::Mat>& weights, std::vector<cv::Mat>& biases, cv::Mat& AL);

	//! computing the cost
	double compute_cost(cv::Mat AL, cv::Mat Y);

	//!< linear backpropagation
	void linear_backward(cv::Mat dZ, cv::Mat A_prev, cv::Mat W, cv::Mat b, cv::Mat& dA_prev, cv::Mat& dW, cv::Mat& db);


private:
	std::vector<cv::Mat> Z;
	std::vector<cv::Mat> A;
	std::vector<cv::Mat> weight_grads;
	std::vector<cv::Mat> bias_grads;

};




#endif
