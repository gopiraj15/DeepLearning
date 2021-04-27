#include "DeepNeuralNetwork.h"

using namespace cv;
using namespace std;


int main(int argc, char **argv)
{

	DeepNeuralNetwork dnn;

	cv::Mat X = (cv::Mat_<double>(3, 2) << -0.41675785, - 0.05626683,
	-2.1361961,   1.64027081,
	-1.79343559, - 0.84174737);

	dnn.initialize_deep_parameters(X, { 5, 4, 3 });

	cv::Mat W = (cv::Mat_<double>(1, 3) << 0.50288142, - 1.24528809, - 1.05795222);
	cv::Mat b = cv::Mat::ones(1, 1, CV_64F) * -0.90900761;

	cv::Mat A;
	dnn.linear_activation_forward(X, W, b, A, "relu");

	cout << A << endl;

	std::vector<Mat> weights, biases;
	//W1
	weights.push_back((cv::Mat_<double>(4, 5) << 
		0.35480861, 1.81259031, -1.3564758, -0.46363197, 0.82465384,
		-1.17643148, 1.56448966, 0.71270509, -0.1810066, 0.53419953,
		-0.58661296, -1.48185327, 0.85724762, 0.94309899, 0.11444143,
		-0.02195668, -2.12714455, -0.83440747, -0.46550831, 0.23371059));
	
	//B1
	biases.push_back((cv::Mat_<double>(4, 1) <<
		1.38503523,
		-0.51962709,
		-0.78015214,
		0.95560959));


	//W2
	weights.push_back((cv::Mat_<double>(3, 4) << 
		-0.12673638, -1.36861282, 1.21848065, -0.85750144,
		-0.56147088, -1.0335199, 0.35877096, 1.07368134,
		-0.37550472, 0.39636757, -0.47144628, 2.33660781));

	//B2
	biases.push_back((cv::Mat_<double>(3, 1) <<
		1.50278553,
		-0.59545972,
		0.52834106));

	//W3
	weights.push_back((cv::Mat_<double>(1, 3) << 0.9398248, 0.42628539, -0.75815703));

	//B3
	biases.push_back((cv::Mat_<double>(1, 1) << -0.16236698));

	X = (cv::Mat_<double>(5, 4) <<
		-0.31178367, 0.72900392, 0.21782079, -0.8990918,
		-2.48678065, 0.91325152, 1.12706373, -1.51409323,
		1.63929108, -0.4298936, 2.63128056, 0.60182225,
		-0.33588161, 1.23773784, 0.11112817, 0.12915125,
		0.07612761, -0.15512816, 0.63422534, 0.810655);


	Mat AL;
	dnn.L_model_forward(X, weights, biases, AL);

	cout << endl << AL << endl;


	cout << "cost: " << dnn.compute_cost(Mat((cv::Mat_<double>(1, 3) << 0.8, 0.9, 0.4)), Mat((cv::Mat_<double>(1, 3) << 1, 1, 0))) << endl;

	return 0;
}