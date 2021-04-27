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

	return 0;
}