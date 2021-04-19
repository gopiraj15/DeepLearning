#ifndef __NEURL_NET_H__
#define __NEURL_NET_H__
#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>


//This function loads the sample data
void load_dataset(std::string x_data, std::string y_data, std::vector<double>& X, std::vector<double>& Y);


//This function computes tanh on the input Mat
cv::Mat tanh(cv::Mat ip);


//computes sigmoid on the input Mat
cv::Mat sigmoid(cv::Mat ip);

//computes the sum of the columns and scales the output with m
cv::Mat col_sum(cv::Mat ip, double m);


class NeuralNetwork
{
public:
	std::vector<cv::Mat> weights; //!< Weights of different layers
	std::vector<cv::Mat> biases;//!< biases of different layers
	int n_x; //!< size of the input layer
	int n_y; //!< size of the output layer
	int n_h; //!< number of hidden layers
	int m; //!< number of training samples
	cv::Mat predictions; //!< predicted outputs

	//! Default constructor
	NeuralNetwork()
	{
		n_x = n_y = n_h = m = 0;
	}

	void layer_sizes(cv::Mat X, cv::Mat Y, int n_hidden)
	{
		n_x = X.rows;
		n_y = Y.rows;
		m = X.cols;
		n_h = n_hidden;

		initialize_parameters(n_x, n_h, n_y);


	}

	//!< This function initializes the parameters for the neural network
	void initialize_parameters(const int& nx, const int& nh, const int& ny)
	{
		
		weights.emplace_back(cv::Mat::ones(n_h, n_x, CV_64FC1));
		weights.emplace_back(cv::Mat::ones(n_y, n_h, CV_64FC1));

		biases.emplace_back(cv::Mat::zeros(n_h, 1, CV_64FC1));
		biases.emplace_back(cv::Mat::zeros(n_y, 1, CV_64FC1));

		//Random Initialization of weights
		randn(weights[0], cv::Scalar(0), cv::Scalar(0.01 * m / 2));
		randn(weights[1], cv::Scalar(0), cv::Scalar(0.01 * m / 2));

		Z.resize(weights.size());
		A.resize(weights.size());

		weight_grads.resize(weights.size());
		bias_grads.resize(weights.size());


	}

	//! This function performs one iteration of forward propagation
	void forward_propagation(cv::Mat X)
	{
		//broadcasting the bias values
		cv::Mat b1(biases[0].size(), CV_64FC(X.cols));
		std::vector<cv::Mat> tmpB;
		for (int i = 0; i < X.cols; ++i)
		{
			tmpB.emplace_back(biases[0]);
		}
		cv::merge(tmpB, b1); tmpB.clear();

		Z[0] = weights[0] * X + b1.reshape(1, biases[0].rows);
		A[0] = tanh(Z[0]);

		//broadcasting the bias values
		cv::Mat b2(biases[1].size(), CV_64FC(X.cols));
		for (int i = 0; i < X.cols; ++i)
		{
			tmpB.emplace_back(biases[1]);
		}
		cv::merge(tmpB, b2);

		Z[1] = weights[1] * A[0] + b2.reshape(1, biases[1].rows);
		A[1] = sigmoid(Z[1]);

	}

	//!< This function computes the cost
	double compute_cost(cv::Mat Y)
	{

		cv::Mat p_A2, p_m_A2;
		cv::log(A[1], p_A2); cv::log(1 - A[1], p_m_A2);

		cv::Mat logprobs = p_A2.mul(Y) + p_m_A2.mul(1 - Y);
		return cv::sum(logprobs)[0] / (-m);

	}

	//!< this function performs one iteration of backpropagation
	void backward_propagation(cv::Mat X, cv::Mat Y)
	{
		
		cv::Mat dZ2 = A[1] - Y;
		cv::Mat dW2 = (dZ2 * A[0].t()) / m;
		cv::Mat db2 = col_sum(dZ2, m);

		

		cv::Mat A1p2; cv::pow(A[0], 2, A1p2);
		cv::Mat dZ1 = (weights[1].t() * dZ2).mul(1 - A1p2);
		cv::Mat dW1 = (dZ1 * X.t()) / m;
		cv::Mat db1 = col_sum(dZ1, m);

		weight_grads[0] = (dW1);
		weight_grads[1] = (dW2);
		bias_grads[0] = (db1);
		bias_grads[1] = (db2);

	}

	void printWeights()
	{
		for (int i = 0; i < weights.size(); ++i)
		{
			std::cout << "W" << i + 1 << ": " << std::endl << weights[i] << std::endl;
			std::cout << "B" << i + 1 << ": " << std::endl << biases[i] << std::endl << std::endl;
		}
	}

	//!< this function updates the parameters using the computer gradients
	void update_parameters(const double& learning_rate = 1.2)
	{
		
		weights[0] -= learning_rate * weight_grads[0];
		weights[1] -= learning_rate * weight_grads[1];

		biases[0] -= learning_rate * bias_grads[0];
		biases[1] -= learning_rate * bias_grads[1];

	}


	//! Constructor that trains the Neural network on the input
	NeuralNetwork(cv::Mat X, cv::Mat Y, const int& n_h, const int& num_iterations = 10000, const bool& print_cost = true)
	{
		//Initializing the parameters
		layer_sizes(X, Y, n_h);

		for (int i = 0; i < num_iterations; ++i)
		{
			//performing forward propagation
			forward_propagation(X);

			//computing the cost
			double cost = compute_cost(Y);

			//performing backpropagation to calculate the gradients
			backward_propagation(X, Y);

			//updating the parameters
			update_parameters();

			if (print_cost && i % 1000 == 0)
				std::cout << "Cost after iteration " << i << ": " << cost << std::endl;

		}

	}

	void predict(cv::Mat X)
	{
		forward_propagation(X);
		predictions = cv::Mat::zeros(1, A[1].cols, CV_64F);

		for (int i = 0; i < A[1].cols; ++i)
		{
			if (A[1].at<double>(i) > 0.5)
			{
				predictions.at<double>(i) = 1;
			}
			else
			{
				predictions.at<double>(i) = 0;
			}
		}

	}



private:
	std::vector<cv::Mat> Z;
	std::vector<cv::Mat> A;
	std::vector<cv::Mat> weight_grads;
	std::vector<cv::Mat> bias_grads;

};


#endif
