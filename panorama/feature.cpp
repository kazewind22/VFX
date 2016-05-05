#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>
#include <array>

using namespace cv;
using namespace std;

int THRESH = 100000;

typedef struct feat
{
	int num = 0;
	vector<tuple<int,int>> keypoints;
	vector<array<int,128>> descriptors;
} Feat;

void loadImageSeq(string path, vector<Mat> &images, vector<float> &focalLengths);
void cylindricalWarping(const Mat &src, Mat &dst, Mat &mask, Feat &feat, float f);
void gradI(const Mat &src, Mat &Ix, Mat &Iy, Mat &Io);
double ResponseFunction(const Mat &M, const double k);
void featureDescriptor(const vector<tuple<int, int>> &keypoints, const Mat &Io, vector<array<int,128>> &descriptors);
void getFeatures(const Mat &img, Feat &feat);
double cosineSimilarity(const array<int, 128> des1, const array<int, 128> des2);
void featureMatching(const Feat &feat1, const Feat &feat2, vector<array<int,2>> &matchs);
void combine2Images(const Mat &src1, const Mat &src2, Mat &dst);

int main(int argc, char** argv)
{
	string path = argv[1];
	vector<Mat> images;
	vector<float> focalLengths;
	vector<Mat> warped_imgs;
	vector<Mat> masks;
	vector<Feat> feats;

	loadImageSeq(path, images, focalLengths);

	for(int i = 0; i < images.size(); i++)
	{
		Feat feat;
		getFeatures(images[i], feat);
		feats.push_back(feat);
	}

	Mat img1 = images[1].clone();
	Mat img2 = images[2].clone();

	for(int index = 0; index < feats[1].num; index++)
	{
		int i = get<0>(feats[1].keypoints[index]);
		int j = get<1>(feats[1].keypoints[index]);
		circle(img1,Point(j,i),2,Scalar::all(22));
	}

	for(int index = 0; index < feats[2].num; index++)
	{
		int i = get<0>(feats[2].keypoints[index]);
		int j = get<1>(feats[2].keypoints[index]);
		circle(img2,Point(j,i),2,Scalar::all(22));
	}

	vector<array<int,2>> matchs;
	featureMatching(feats[1], feats[2], matchs);

	Mat match;
	combine2Images(img2,img1,match);

	for(int i = 0; i < matchs.size(); i++)
	{
		int x1 = get<1>(feats[1].keypoints[matchs[i][0]])+img2.cols;
		int y1 = get<0>(feats[1].keypoints[matchs[i][0]]);
		int x2 = get<1>(feats[2].keypoints[matchs[i][1]]);
		int y2 = get<0>(feats[2].keypoints[matchs[i][1]]);
		line(match, Point(x1, y1), Point(x2, y2), Scalar(rand()%256,rand()%256,rand()%256));
	}

	namedWindow("match_origin", CV_WINDOW_AUTOSIZE);
	imshow("match_origin", match);

	for(int i = 0; i < images.size(); i++)
	{
		Mat cylindrical;
		Mat mask;
		cylindricalWarping(images[i], cylindrical, mask, feats[i], focalLengths[i]);
		warped_imgs.push_back(cylindrical);
		masks.push_back(mask);
	}

	for(int index = 0; index < feats[1].num; index++)
	{
		int i = get<0>(feats[1].keypoints[index]);
		int j = get<1>(feats[1].keypoints[index]);
		circle(warped_imgs[1],Point(j,i),2,Scalar::all(22));
	}

	for(int index = 0; index < feats[2].num; index++)
	{
		int i = get<0>(feats[2].keypoints[index]);
		int j = get<1>(feats[2].keypoints[index]);
		circle(warped_imgs[2],Point(j,i),2,Scalar::all(22));
	}

	Mat match_warp;
	combine2Images(warped_imgs[2],warped_imgs[1],match_warp);

	for(int i = 0; i < matchs.size(); i++)
	{
		int x1 = get<1>(feats[1].keypoints[matchs[i][0]])+img2.cols;
		int y1 = get<0>(feats[1].keypoints[matchs[i][0]]);
		int x2 = get<1>(feats[2].keypoints[matchs[i][1]]);
		int y2 = get<0>(feats[2].keypoints[matchs[i][1]]);
		line(match_warp, Point(x1, y1), Point(x2, y2), Scalar(rand()%256,rand()%256,rand()%256));
	}
	namedWindow("match_warp", CV_WINDOW_AUTOSIZE);
	imshow("match_warp", match_warp);

	waitKey(0);
	return 0;
}

void loadImageSeq(string path, vector<Mat> &images, vector<float> &times)
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

void cylindricalWarping(const Mat &src, Mat &dst, Mat &mask, Feat &feat, float f)
{
	Mat result(src.rows, src.cols, src.type(), Scalar::all(0));
	mask = Mat(src.rows, src.cols, CV_8UC1, Scalar::all(255));
	int xc = src.cols/2;
	int yc = src.rows/2;
	for(int y = 0; y < src.rows; y++)
		for(int x = 0; x < src.cols; x++)
		{
			int x_ = x - xc + 1;
			int y_ = y - yc + 1;
			//cout << "x_: " << x_ << ", y_: " << y_ << endl;
			y_ = y_ * sqrt(1+ pow(tan(x_/f),2));
			x_ = f*tan(x_/f);
			//cout << "x_: " << x_ << ", y_: " << y_ << ", f: " << f << endl;
			x_ += xc - 1;
			y_ += yc - 1;
			if(x_ >= 0.0 && x_ < src.cols && y_ >= 0.0 && y_ < src.rows)
				result.at<Vec3b>(y, x) = src.at<Vec3b>(y_, x_);
			else
			{
				for(int i = -2; i <= 2; i++)
				{
					if(x+i < 0 || x+i > src.cols)
						continue;
					for(int j = -2; j <= 2; j++)
					{
						if(y+j < 0 || y+j > src.rows)
							continue;
						mask.at<uchar>(y+j, x+i) = 0;
					}
				}
			}
		}
	dst = result;

	for(int index = 0; index < feat.keypoints.size(); index++)
	{
		int x = get<1>(feat.keypoints[index]) - xc + 1;
		int y = get<0>(feat.keypoints[index]) - yc + 1;
		y = f * y / sqrt(x*x+f*f);
		x = f * atan((float)x/f);
		float at = fastAtan2((float)x,f);
		x += xc - 1;
		y += yc - 1;
		feat.keypoints[index] = make_tuple(y,x);
	}
}

void gradI(const Mat &src, Mat &Ix, Mat &Iy, Mat &Io)
{
	Mat kernelX(1, 3, CV_64F);
	kernelX.at<double>(0,0) = -1.0f;
	kernelX.at<double>(0,1) = 0.0f;
	kernelX.at<double>(0,2) = 1.0f;

	filter2D(src, Ix, CV_64F, kernelX);

	Mat kernelY(3, 1, CV_64F);
	kernelY.at<double>(0,0) = -1.0f;
	kernelY.at<double>(1,0) = 0.0f;
	kernelY.at<double>(2,0) = 1.0f;

	filter2D(src, Iy, CV_64F, kernelY);

	Mat ori(src.size(), CV_64F);
	for(int i = 0; i < src.rows; i++)
		for(int j = 0; j < src.cols; j++)
			ori.at<double>(i,j) = fastAtan2(Iy.at<double>(i,j), Ix.at<double>(i,j));
	Io = ori;
}

double ResponseFunction(const Mat &M, const double k)
{
	double A = M.at<double>(0,0);
	double B = M.at<double>(1,1);
	double C = M.at<double>(0,1);
	return A * B - C * C - k * (A+B) * (A+B);
}

void featureDescriptor(const vector<tuple<int, int>> &keypoints, const Mat &Io, vector<array<int,128>> &descriptors)
{
	descriptors.clear();
	cout << keypoints.size() << endl;
	for(int index = 0; index < keypoints.size(); index++)
	{
		int y = get<0>(keypoints[index]);
		int x = get<1>(keypoints[index]);
		array<int, 128> count={0};
		int block[4] = {-8, -4, 1, 5};
		for(int by = 0; by < 4; by++)
		{
			int y_ = y + block[by];
			for(int bx = 0; bx < 4; bx++)
			{
				int x_ = x + block[bx];
				for(int i = 0; i < 4; i++)
				{
					for(int j = 0; j < 4; j++)
					{
						count[8*(4*by+bx)+floor(Io.at<double>(y_+i,x_+j) / 45)]++;
					}
				}
			}
		}
		descriptors.push_back(count);
	}
}

void getFeatures(const Mat &img, Feat &feat)
{
	Mat gimg;
	cvtColor(img, gimg, COLOR_RGB2GRAY);

	Mat Ix, Iy, Io;
	gradI(gimg, Ix, Iy, Io);

	Mat A, B, C;
	GaussianBlur(Ix.mul(Ix), A, Size(5,5), 1);
	GaussianBlur(Iy.mul(Iy), B, Size(5,5), 1);
	GaussianBlur(Ix.mul(Iy), C, Size(5,5), 1);

	Mat R(img.size(), CV_64F);

	for(int i = 0; i < img.rows; i++)
		for(int j = 0; j < img.cols; j++)
		{
			double m[2][2] = {{A.at<double>(i,j), C.at<double>(i,j)},{C.at<double>(i,j), B.at<double>(i,j)}};
			Mat M = Mat(2,2,CV_64F,m);

			R.at<double>(i,j) = ResponseFunction(M,0.04);
		}

	feat.keypoints.clear();
	for(int i = 9; i < img.rows-9; i++)
		for(int j = 9; j < img.cols-9; j++)
		{
			if(R.at<double>(i,j) > R.at<double>(i-1,j) &&
			   R.at<double>(i,j) > R.at<double>(i+1,j) &&
			   R.at<double>(i,j) > R.at<double>(i,j-1) &&
			   R.at<double>(i,j) > R.at<double>(i,j+1) &&
			   R.at<double>(i,j) > THRESH)
			{
				feat.keypoints.push_back(make_tuple(i,j));
				feat.num++;
			}
		}

	featureDescriptor(feat.keypoints, Io, feat.descriptors);
}

double cosineSimilarity(const array<int, 128> des1, const array<int, 128> des2)
{
	double sum = 0, len1 = 0, len2 = 0;
	for(int i = 0; i < 128; i++)
	{
		sum += des1[i] * des2[i];
		len1 += des1[i] * des1[i];
		len2 += des2[i] * des2[i];
	}
	len1 = sqrt(len1);
	len2 = sqrt(len2);
	return sum/(len1*len2);
}

void featureMatching(const Feat &feat1, const Feat &feat2, vector<array<int,2>> &matchs)
{
	for(int i = 0; i < feat1.num; i++)
	{
		double max_score = -1;
		int max_index = -1;
		for(int j = 0; j < feat2.num; j++)
		{
			double score = cosineSimilarity(feat1.descriptors[i], feat2.descriptors[j]);
			if(score > max_score)
			{
				max_score = score;
				max_index = j;
			}
		}
		if(max_score > 0.85)
		{
			array<int,2> match = {i, max_index};
			matchs.push_back(match);
		}
	}
}

void combine2Images(const Mat &src1, const Mat &src2, Mat &dst)
{
	Mat M(max(src1.rows, src2.rows), src1.cols+src2.cols, CV_8UC3, Scalar::all(0));
	Mat left(M, Rect(0, 0, src1.cols, src1.rows)); // Copy constructor
	src1.copyTo(left);
	Mat right(M, Rect(src1.cols, 0, src2.cols, src2.rows)); // Copy constructor
	src2.copyTo(right);
	dst = M;
}
