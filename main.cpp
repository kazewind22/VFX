#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;

int main(int argc, char** argv) {

	std::string filename = argv[1];
	Mat A;
	A = imread(filename);
	return 0;
}
