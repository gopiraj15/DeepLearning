#include "NeuralNetwork.h"

using namespace cv;
using namespace std;

int main()
{

	vector<double> X_ip, Y_ip;

	//Loading the input data
	load_dataset("data/x_gaussian_quantiles.txt", "data/y_gaussian_quantiles.txt", X_ip, Y_ip);

	cv::Mat X(2, 400, CV_64F, X_ip.data());
	cv::Mat Y(1, 400, CV_64F, Y_ip.data());

	//fitting the model to data
	NeuralNetwork NN(X, Y, 5, 10000, true);

	NN.printWeights();

	NN.predict(X);

	//computing accuracy
	cout << "Accuracy: " << 100 * (Y * NN.predictions.t() + (1 - Y) * (1 - NN.predictions.t())) / NN.m << "%" << endl;

	
	return 0;
}