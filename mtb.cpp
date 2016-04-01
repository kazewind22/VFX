#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>

#define TOLERANCE 6
#define MAX_SHIFT_BIT 7
#define MAX_ROTATE_ANGEL 3

using namespace cv;
using namespace std;

void rotate(const Mat &src, double angle, Mat &dst);
void loadExposureSeq(string path, vector<Mat> &images, vector<float> &times);
void MTBalignement(const vector<Mat> &images, vector<Mat> &aligned_images);
void ImageShrink(const Mat& img, Mat& ret);
void ComputeBitmaps(const Mat& img, Mat& tb, Mat& eb);
void BitmapShift(const Mat& bm, int x, int y, Mat& bm_ret);
int BitmapTotal(const Mat& bm);
double GetExpShift(const Mat& img1, const Mat& img2, double angle, int shift_bits, int shift_ret[2]);
void AlignedImages(const vector<Mat> &images, vector<Mat> &aligned_images, vector<tuple<int,int,double>> &shifts);
void writeAlignedImages(string path, vector<Mat> &imgs);

int main(int argc, char** argv)
{
	string path = argv[1];

	vector<Mat> images;
	vector<Mat> aligned_images;
	vector<float> times;

	loadExposureSeq(path, images, times);

	MTBalignement(images, aligned_images);

	writeAlignedImages(path, aligned_images);

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

void MTBalignement(const vector<Mat> &images, vector<Mat> &aligned_images)
{
	vector<Mat> gray_images;
	vector<Mat> _aligned_images;
	for(int i = 0; i < images.size(); i++)
	{
		Mat gray;
		cvtColor(images[i], gray, COLOR_RGB2GRAY);
		gray_images.push_back(gray);
		Mat tb,eb;
		ComputeBitmaps(gray, tb, eb);
	}

	vector<tuple<int,int,double>> shifts;
	shifts.push_back(make_tuple(0,0,0));
	for(int i = 1; i < gray_images.size(); i++)
	{
		int shift_ret[2];
		double min_err = gray_images[0].rows*gray_images[0].cols;
		int min_shift[2];
		double min_angle;
		for(double angle = -MAX_ROTATE_ANGEL; angle <= MAX_ROTATE_ANGEL; angle += 0.5)
		{
			double err = GetExpShift(gray_images[i-1], gray_images[i], angle, MAX_SHIFT_BIT, shift_ret);
			if(err < min_err)
			{
				min_err = err;
				min_shift[0] = shift_ret[0];
				min_shift[1] = shift_ret[1];
				min_angle = angle;
			}
		}
		shifts.push_back(make_tuple(min_shift[0]+get<0>(shifts[i-1]),min_shift[1]+get<1>(shifts[i-1]),min_angle+get<2>(shifts[i-1])));
	}

	AlignedImages(images, _aligned_images, shifts);

	aligned_images = _aligned_images;
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
	int index = 0;
	vector<int> pixels(img.rows*img.cols);
	MatConstIterator_<uchar> it = img.begin<uchar>(), it_end = img.end<uchar>();
	for(; it != it_end; ++it)
	{
		sum += *it;
		pixels[index] = *it;
		index++;
	}

	sort(pixels.begin(), pixels.end());
	double threshold = pixels[img.rows*img.cols*3/4];

	Mat tb_(img.size(),CV_8UC1,Scalar(0));
	Mat eb_(img.size(),CV_8UC1,Scalar(1));

	for(int i = 0; i < img.rows; i++)
		for(int j = 0; j < img.cols; j++)
		{
			double pixel = img.at<uchar>(i,j);
			if(pixel >= threshold)
				tb_.at<uchar>(i,j) = 1;
			if(fabs(pixel - threshold) < TOLERANCE)
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

double GetExpShift(const Mat& img1, const Mat& img2, double angle, int shift_bits, int shift_ret[2])
{
	double min_err;
	int cur_shift[2];
	Mat tb1, tb2;
	Mat eb1, eb2;

	if(shift_bits > 0)
	{
		Mat sml_img1, sml_img2;
		ImageShrink(img1, sml_img1);
		ImageShrink(img2, sml_img2);
		GetExpShift(sml_img1, sml_img2, angle, shift_bits-1, cur_shift);
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
			Mat rotated_tb2(img1.size(), CV_8UC1, Scalar(0));
			Mat rotated_eb2(img1.size(), CV_8UC1, Scalar(0));
			Mat diff_b,eb;
			BitmapShift(tb2, xs, ys, shifted_tb2);
			BitmapShift(eb2, xs, ys, shifted_eb2);
			rotate(shifted_tb2, angle, rotated_tb2);
			rotate(shifted_eb2, angle, rotated_eb2);
			bitwise_xor(tb1, rotated_tb2, diff_b);
			bitwise_and(eb1, rotated_eb2, eb);
			bitwise_and(diff_b, eb, diff_b);
			double err = (double)BitmapTotal(diff_b);
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

	return min_err;
}

void AlignedImages(const vector<Mat> &images, vector<Mat> &aligned_images, vector<tuple<int,int,double>> &shifts)
{
	vector<Mat> images_;
	for(int i = 0; i < images.size(); i++)
	{
		if (get<2>(shifts[i])!=0)
		{
			Mat rotated;
			rotate(images[i], get<2>(shifts[i]), rotated);
			images_.push_back(rotated);
		}
		else
			images_.push_back(images[i]);
	}

	Size size = images_[0].size();
	int sum_x = 0, sum_y = 0;
	for(int i = 0; i < images_.size(); i++)
	{
		sum_x += get<0>(shifts[i]);
		sum_y += get<1>(shifts[i]);
	}
	int center_x = copysign(fabs(sum_x) / images_.size(), sum_x);
	int center_y = copysign(fabs(sum_y) / images_.size(), sum_y);
	for(int i = 0; i < images_.size(); i++)
	{
		get<0>(shifts[i]) -= center_x;
		get<1>(shifts[i]) -= center_y;
	}
	int left = 0, right = 0, top = 0, bottom = 0;
	for(int i = 0; i < images_.size(); i++)
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
	for(int i = 0; i < images_.size(); i++)
	{
		Rect roi = Rect(left-get<0>(shifts[i]), top+get<1>(shifts[i]),0,0) + size;
		cout << roi << endl;
		aligned_images.push_back(images_[i](roi));
	}
	return;
}

void writeAlignedImages(string path, vector<Mat> &imgs)
{
	path += "/";
	ifstream list_file((path+"list.txt").c_str());
	string name;
	float time;
	int i = 0;
	while(list_file >> name >> time)
	{
		imwrite(path+"aligned_"+name, imgs[i]);
		i++;
	}
	list_file.close();
	return;
}

void rotate(const Mat &src, double angle, Mat &dst)
{
	Point2f pt(src.cols/2., src.rows/2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
}
