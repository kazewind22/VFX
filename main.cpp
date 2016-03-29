#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>

#define TOLERANCE 6
#define MAX_SHIFT_BIT 4

using namespace cv;
using namespace std;

void loadExposureSeq(string path, vector<Mat> &images, vector<float> &times);
void ImageShrink(const Mat& img, Mat& ret);
void ComputeBitmaps(const Mat& img, Mat& tb, Mat& eb);
void BitmapShift(const Mat& bm, int x, int y, Mat& bm_ret);
int BitmapTotal(const Mat& bm);
void GetExpShift(const Mat& img1, const Mat& img2, int shift_bits, int shift_ret[2]);
void LocateImages(const vector<Mat> &images, vector<Mat> &located_images, vector<tuple<int,int>> &shifts);
void writeLocateImages(string path, vector<Mat> &imgs);

int main(int argc, char** argv)
{
	string path = argv[1];

	vector<Mat> images;
	vector<Mat> g_images;
	vector<Mat> l_images;
	vector<float> times;

	loadExposureSeq(path, images, times);

	for(int i = 0; i < images.size(); i++)
	{
		Mat gray;
		cvtColor(images[i], gray, COLOR_RGB2GRAY);
		g_images.push_back(gray);
	}

	vector<tuple<int,int>> shifts;
	shifts.push_back(make_tuple(0,0));
	for(int i = 1; i < g_images.size(); i++)
	{
		int shift_ret[2];
		GetExpShift(g_images[0], g_images[i], MAX_SHIFT_BIT, shift_ret);
		shifts.push_back(make_tuple(shift_ret[0],shift_ret[1]));
	}

	LocateImages(images, l_images, shifts);

	writeLocateImages(path, l_images);

	return 0;
}

void loadExposureSeq(string path, vector<Mat> &images, vector<float> &times)
{
	path += "/";
	ifstream list_file((path+"list.txt").c_str());
	string name;
	float time;
	while(list_file >> name >> time)
	{
		Mat image = imread(path+name);
		images.push_back(image);
		times.push_back(time);
	}
	list_file.close();
	return;
}

void ImageShrink(const Mat& img, Mat& ret)
{
	Mat ret_(img.size()/2, CV_8UC1, Scalar(0));
	for(int i = 0; i < ret_.rows; i++)
		for(int j = 0; j < ret_.cols; j++)
		{
			int sum = img.at<uchar>(2*i,2*j)+img.at<uchar>(2*i+1,2*j)+
				      img.at<uchar>(2*i,2*j+1)+img.at<uchar>(2*i+1,2*j+1);
			ret_.at<uchar>(i,j) = sum / 4;
		}
	ret = ret_;
	return;
}

void ComputeBitmaps(const Mat& img, Mat& tb, Mat& eb)
{
	double sum = 0;
	MatConstIterator_<uchar> it = img.begin<uchar>(), it_end = img.end<uchar>();
	for(; it != it_end; ++it)
		sum += *it;
	double mid = sum / img.rows / img.cols;

	Mat tb_(img.size(),CV_8UC1,Scalar(0));
	Mat eb_(img.size(),CV_8UC1,Scalar(1));

	for(int i = 0; i < img.rows; i++)
		for(int j = 0; j < img.cols; j++)
		{
			double pixel = img.at<uchar>(i,j);
			if(pixel >= mid)
				tb_.at<uchar>(i,j) = 1;
			if(fabs(pixel - mid) < TOLERANCE)
				eb_.at<uchar>(i,j) = 0;
		}
	tb = tb_;
	eb = eb_;
	return;
}

void BitmapShift(const Mat& bm, int x, int y, Mat& bm_ret)
{
	// error checking
	assert(fabs(x) < bm.cols && fabs(y) < bm.rows);

	// first create a border around the parts of the Mat that will be exposed
	int t = 0, b = 0, l = 0, r = 0;
	if (x > 0) l =  x;
	if (x < 0) r = -x;
	if (y > 0) t =  y;
	if (y < 0) b = -y;
	Mat padded;
	copyMakeBorder(bm, padded, t, b, l, r, BORDER_CONSTANT, Scalar(0));

	// construct the region of interest around the new matrix
	Rect roi = Rect(max(-x,0),max(-y,0),0,0) + bm.size();

	bm_ret = padded(roi);
	return;
}

int BitmapTotal(const Mat& bm)
{
	int count = 0;
	for(int i = 0; i < bm.rows; i++)
		for(int j = 0; j < bm.cols; j++)
			count += bm.at<uchar>(i,j);
	return count;
}

void GetExpShift(const Mat& img1, const Mat& img2, int shift_bits, int shift_ret[2])
{
	int min_err;
	int cur_shift[2];
	Mat tb1, tb2;
	Mat eb1, eb2;

	if(shift_bits > 0)
	{
		Mat sml_img1, sml_img2;
		ImageShrink(img1, sml_img1);
		ImageShrink(img2, sml_img2);
		GetExpShift(sml_img1, sml_img2, shift_bits-1, cur_shift);
		sml_img1.release();
		sml_img2.release();
		cur_shift[0] *= 2;
		cur_shift[1] *= 2;
	}
	else
	{
		cur_shift[0] = 0;
		cur_shift[1] = 0;
	}
	ComputeBitmaps(img1, tb1, eb1);
	ComputeBitmaps(img2, tb2, eb2);
	min_err = img1.rows * img1.cols;
	for(int i = -1; i <= 1; i++)
		for(int j = -1; j <= 1; j++)
		{
			int xs = cur_shift[0]+i;
			int ys = cur_shift[1]+j;
			Mat shifted_tb2(img1.size(), CV_8UC1, Scalar(0));
			Mat shifted_eb2(img1.size(), CV_8UC1, Scalar(0));
			Mat diff_b;
			BitmapShift(tb2, xs, ys, shifted_tb2);
			BitmapShift(eb2, xs, ys, shifted_eb2);
			bitwise_xor(tb1, shifted_tb2, diff_b);
			bitwise_and(diff_b, eb1, diff_b);
			bitwise_and(diff_b, shifted_eb2, diff_b);
			int err = BitmapTotal(diff_b);
			if(err < min_err)
			{
				shift_ret[0] = xs;
				shift_ret[1] = ys;
				min_err = err;
			}
			shifted_tb2.release();
			shifted_eb2.release();
		}
	tb1.release();
	eb1.release();
	tb2.release();
	eb2.release();

	return;
}

void LocateImages(const vector<Mat> &images, vector<Mat> &located_images, vector<tuple<int,int>> &shifts)
{
	Size size = images[0].size();
	int sum_x = 0, sum_y = 0;
	for(int i = 0; i < images.size(); i++)
	{
		sum_x += get<0>(shifts[i]);
		sum_y += get<1>(shifts[i]);
	}
	int center_x = copysign(fabs(sum_x) / images.size(), sum_x);
	int center_y = copysign(fabs(sum_y) / images.size(), sum_y);
	for(int i = 0; i < images.size(); i++)
	{
		get<0>(shifts[i]) -= center_x;
		get<1>(shifts[i]) -= center_y;
	}
	int left = 0, right = 0, top = 0, bottom = 0;
	for(int i = 0; i < images.size(); i++)
	{
		if(get<0>(shifts[i]) >= 0)
			left = max(left, get<0>(shifts[i]));
		else
			right = max(right, -get<0>(shifts[i]));
		if(get<1>(shifts[i]) >= 0)
			bottom = max(bottom, get<1>(shifts[i]));
		else
			top = max(top, -get<1>(shifts[i]));
	}
	size.width -= left+right;
	size.height -= top+bottom;
	for(int i = 0; i < images.size(); i++)
	{
		Rect roi = Rect(left-get<0>(shifts[i]), top+get<1>(shifts[i]),0,0) + size;
		located_images.push_back(images[i](roi));
	}
	return;
}

void writeLocateImages(string path, vector<Mat> &imgs)
{
	path += "/";
	ifstream list_file((path+"list.txt").c_str());
	string name;
	float time;
	int i = 0;
	while(list_file >> name >> time)
	{
		imwrite(path+"located_"+name, imgs[i]);
		i++;
	}
	list_file.close();
	return;
}
