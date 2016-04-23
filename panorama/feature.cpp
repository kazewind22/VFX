#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>

using namespace cv;
using namespace std;

void gradI(const Mat &src, Mat &Ix, Mat &Iy);
double ResponseFunction(const Mat &M, const double k);

int main(int argc, char** argv)
{
	string path = argv[1];
	Mat img = imread(path);

	Mat gimg;
	cvtColor(img, gimg, COLOR_RGB2GRAY);

	Mat Ix, Iy;
	gradI(gimg, Ix, Iy);

	Mat A, B, C;
	GaussianBlur(Ix.mul(Ix), A, Size(5,5), 1);
	GaussianBlur(Iy.mul(Iy), B, Size(5,5), 1);
	GaussianBlur(Ix.mul(Iy), C, Size(5,5), 1);

	Mat R(img.size(), CV_64F, Scalar::all(0));

	for(int i = 0; i < img.rows; i++)
		for(int j = 0; j < img.cols; j++)
		{
			double m[2][2] = {{A.at<double>(i,j), C.at<double>(i,j)},{C.at<double>(i,j), B.at<double>(i,j)}};
			Mat M = Mat(2,2,CV_64F,m);

			R.at<double>(i,j) = ResponseFunction(M,0.04);
		}

	for(int i = 1; i < img.rows-1; i++)
		for(int j = 0; j < img.cols-1; j++)
		{
			if(R.at<double>(i,j) > R.at<double>(i-1,j) &&
			   R.at<double>(i,j) > R.at<double>(i+1,j) &&
			   R.at<double>(i,j) > R.at<double>(i,j-1) &&
			   R.at<double>(i,j) > R.at<double>(i,j+1) &&
			   R.at<double>(i,j) > 500000)
				circle(img,Point(j,i),3,Scalar(22));
		}

	imshow("img",img);

	waitKey(0);
	return 0;
}

void gradI(const Mat &src, Mat &Ix, Mat &Iy)
{
	cv::Mat kernelX(1, 3, CV_64F);
	kernelX.at<double>(0,0) = -1.0f;
	kernelX.at<double>(0,1) = 0.0f;
	kernelX.at<double>(0,2) = 1.0f;

	filter2D(src, Ix, CV_64F, kernelX);

	cv::Mat kernelY(3, 1, CV_64F);
	kernelY.at<double>(0,0) = -1.0f;
	kernelY.at<double>(1,0) = 0.0f;
	kernelY.at<double>(2,0) = 1.0f;

	filter2D(src, Iy, CV_64F, kernelY);
}

double ResponseFunction(const Mat &M, const double k)
{
	double A = M.at<double>(0,0);
	double B = M.at<double>(1,1);
	double C = M.at<double>(0,1);
	return A * B - C * C - k * (A+B);
}
