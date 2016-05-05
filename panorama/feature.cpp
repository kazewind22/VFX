#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>
#include <array>
#include <cstdio>

using namespace cv;
using namespace std;

int original_y;
int final_y;

typedef struct feat
{
	int num = 0;
	vector<tuple<int,int>> keypoints;
	vector<array<int,128>> descriptors;
} Feat;

void loadImageSeq(string path, vector<Mat> &images, vector<float> &focalLengths);
void cylindricalWarping(const Mat &src, Mat &dst, float f);
void gradI(const Mat &src, Mat &Ix, Mat &Iy, Mat &Io);
double ResponseFunction(const Mat &M, const double k);
void featureDescriptor(const vector<tuple<int, int>> &keypoints, const Mat &Io, vector<array<int,128>> &descriptors);
void getFeatures(const Mat &img, Feat &feat);
double cosineSimilarity(const array<int, 128> des1, const array<int, 128> des2);
double cosineSimilarity(const tuple<int, int> v1, const tuple<int, int> v2);
void featureMatching(const Feat &feat1, const Feat &feat2, vector<array<int,2>> &matchs);
void combine2Images(const Mat &src1, const Mat &src2, Mat &dst);
void detectOutliers(const int offset, const Feat &feat1, const Feat &feat2, const int width, const vector<array<int,2>> &matchs, vector<array<int,2>> &puredMatchs);
void stitchImages(const Mat &src1, const Mat &src2, const Mat &M, Mat &dst);

int main(int argc, char** argv)
{
	string path = "parrington";
	vector<Mat> images;
	vector<float> focalLengths;
	vector<Mat> warped_imgs;

	loadImageSeq(path, images, focalLengths);

	for(int i = 0; i < images.size(); i++)
	{
		Mat cylindrical;
		cylindricalWarping(images[i], cylindrical, focalLengths[i]);
		warped_imgs.push_back(cylindrical);
		// ostringstream w;
		// w << i <<"_warp.jpg";
		// string wName= w.str();
		// imwrite(wName,warped_imgs[warped_imgs.size()-1]);
	}

	Feat* f1 = new Feat;
	Feat* f2 = new Feat;
	Feat feat1 = *f1;
	Feat feat2 = *f2;


	// images[0] = warped_imgs[0];
	// images[1] = warped_imgs[1];
	Mat stitchedImage;
	Mat image;
	Mat image0;
	Mat image1;
	Mat _M(3,3,CV_64FC1,Scalar::all(0));
	_M.at<double>(0,0)=1;
	_M.at<double>(1,1)=1;
	_M.at<double>(2,2)=1;
	original_y = warped_imgs[images.size()-1].rows;
	for(int imgIndex = images.size()-1; imgIndex >= 1; imgIndex--){
		if(imgIndex == images.size()-1)
			image = warped_imgs[imgIndex];
		else
			image = stitchedImage;
		cout << images.size()-imgIndex << "th iter" << endl;

		image1 = warped_imgs[imgIndex];
		image0 = warped_imgs[imgIndex-1];				

		cout << "get features" << endl;
		getFeatures(image0, feat1);
		getFeatures(image1, feat2);

		for(int index = 0; index < feat1.num; index++)
		{
			int i = get<0>(feat1.keypoints[index]);
			int j = get<1>(feat1.keypoints[index]);
			// circle(image0,Point(j,i),2,Scalar(22));
		}

		for(int index = 0; index < feat2.num; index++)
		{
			int i = get<0>(feat2.keypoints[index]);
			int j = get<1>(feat2.keypoints[index]);
			// circle(image1,Point(j,i),2,Scalar(22));
		}

		cout << "feature matching" << endl;
		vector<array<int,2>> matchs;
		// matchs.clear();
		featureMatching(feat1, feat2, matchs);


		cout << "detect outliers" << endl;
		vector<array<int,2>> puredMatchs;
		// puredMatchs.clear();
		detectOutliers(image1.cols, feat1, feat2, image0.cols, matchs, puredMatchs);

		Mat match;
		combine2Images(image1,image0,match);
		for(int i = 0; i < puredMatchs.size(); i++)
		{
			int x1 = get<1>(feat1.keypoints[puredMatchs[i][0]])+image1.cols;
			int y1 = get<0>(feat1.keypoints[puredMatchs[i][0]]);
			int x2 = get<1>(feat2.keypoints[puredMatchs[i][1]]);
			int y2 = get<0>(feat2.keypoints[puredMatchs[i][1]]);
			line(match, Point(x1, y1), Point(x2, y2), Scalar(rand()%256,rand()%256,rand()%256));
		}

		ostringstream stringStreamM;
		stringStreamM << imgIndex <<"_iter_match.jpg";
		string nameMatch= stringStreamM.str();
		imwrite(nameMatch,match);

		vector<Point2f> obj;
		vector<Point2f> scene;
		for( int i = 0; i < puredMatchs.size(); i++ )
		{	
			Point2f a(get<1>(feat1.keypoints[puredMatchs[i][0]]), get<0>(feat1.keypoints[puredMatchs[i][0]]));
			Point2f b(get<1>(feat2.keypoints[puredMatchs[i][1]]), get<0>(feat2.keypoints[puredMatchs[i][1]]));
			//-- Get the keypoints from the good matches
			obj.push_back(b);
			scene.push_back(a);
		}

		cout << "transforamtion matrix" << endl;
		Mat objVector = Mat(puredMatchs.size(),3,CV_64F,Scalar::all(0));
		Mat sceneVector = Mat(puredMatchs.size(),3,CV_64F,Scalar::all(0));
		for(int i = 0; i < puredMatchs.size(); i++)
		{
			objVector.at<double>(i,0) = obj[i].x;
			objVector.at<double>(i,1) = obj[i].y;
			objVector.at<double>(i,2) = 1;
			sceneVector.at<double>(i,0) = scene[i].x;
			sceneVector.at<double>(i,1) = scene[i].y;
			sceneVector.at<double>(i,2) = 1;
		}
		
		Mat tmpM;
		solve(sceneVector, objVector, tmpM, DECOMP_NORMAL );
		tmpM = tmpM.t();

		Mat M = _M*tmpM;
		_M = M;
		// cout << "tmpM:" << tmpM << endl;
		// cout << "M:" << M << endl;
		// cout << "H:" << _M << endl;
		
		cout << "image stitching" << endl;
		
		stitchImages(image,image0,M,stitchedImage);
	
	}

	cout << "stitch size: " << stitchedImage.size() << endl;

	ostringstream s;
	s << argv[1] <<"_stitched.jpg";
	string stitchName= s.str();
	imwrite(stitchName,stitchedImage);

	
	imshow("match", stitchedImage);
	waitKey(0);
	return 0;
}

void stitchImages(const Mat &src1, const Mat &src2,const Mat &M, Mat &dst)
{

	// result size
	Mat T = M.inv(DECOMP_LU);
	Mat scr2Index = Mat(src2.rows,src2.cols,CV_32FC2,Scalar::all(0));
	double max_x = 0; 
	double max_y = 0;
	double min_x = 10000; 
	double min_y = 10000;
	for(int i = 0; i < src2.rows; i++)
	{
		for(int j = 0; j < src2.cols; j++)
		{
			scr2Index.at<Vec2f>(i,j) = Vec2f(M.at<double>(0,0)*j+M.at<double>(0,1)*i+M.at<double>(0,2),M.at<double>(1,0)*j+M.at<double>(1,1)*i+M.at<double>(1,2));		
			if(scr2Index.at<Vec2f>(i,j).val[0]>max_x)
				max_x = scr2Index.at<Vec2f>(i,j).val[0];
			if(scr2Index.at<Vec2f>(i,j).val[1]>max_y)
				max_y = scr2Index.at<Vec2f>(i,j).val[1];
			if(scr2Index.at<Vec2f>(i,j).val[0]<min_x)
				min_x = scr2Index.at<Vec2f>(i,j).val[0];
			if(scr2Index.at<Vec2f>(i,j).val[1]<min_y)
				min_y = scr2Index.at<Vec2f>(i,j).val[1];
		}
	}
	final_y = (int)max_y;
	Mat result(max(src1.rows, (int)max_y)+1, (int)max_x+1, CV_8UC3, Scalar::all(0));

	for(int i = 0; i < result.rows; i++)
	{
		for(int j = 0; j < result.cols; j++)
		{	
			//Forward warping not good we have to do inverse warping.
			double y = (T.at<double>(1,0)*j+T.at<double>(1,1)*i+T.at<double>(1,2));
			double x = (T.at<double>(0,0)*j+T.at<double>(0,1)*i+T.at<double>(0,2));
			if(y >= 0 && y < src2.rows && x >= 0 && x < src2.cols)
			{
				double y_1 = floor(y);
				double x_1 = floor(x);
				double y_2 = y_1+1;
				double x_2 = x_1+1;
				if(src2.at<Vec3b>(y_1,x_1).val[0]==0 || src2.at<Vec3b>(y_1,x_2).val[0]==0 
					|| src2.at<Vec3b>(y_2,x_1).val[0]==0 || src2.at<Vec3b>(y_2,x_2).val[0]==0)
				{
					if(src2.at<Vec3b>(y_1,x_1).val[0]!=0)
					{					
						result.at<Vec3b>(i,j) = src2.at<Vec3b>(y_1,x_1);
					}
					else if(src2.at<Vec3b>(y_2,x_1).val[0]!=0)
					{					
						result.at<Vec3b>(i,j) = src2.at<Vec3b>(y_2,x_1);
					}
					else if(src2.at<Vec3b>(y_1,x_2).val[0]!=0)
					{					
						result.at<Vec3b>(i,j) = src2.at<Vec3b>(y_1,x_2);
					}
					else if(src2.at<Vec3b>(y_2,x_2).val[0]!=0)
					{					
						result.at<Vec3b>(i,j) = src2.at<Vec3b>(y_2,x_2);
					}
					else
					{
						result.at<Vec3b>(i,j).val[0] = 0;
						result.at<Vec3b>(i,j).val[1] = 0;
						result.at<Vec3b>(i,j).val[2] = 0;
					}
				}

				else
				{
					result.at<Vec3b>(i,j).val[0] += src2.at<Vec3b>(y_1,x_1).val[0]*(y_2-y)*(x_2-x);
					result.at<Vec3b>(i,j).val[1] += src2.at<Vec3b>(y_1,x_1).val[1]*(y_2-y)*(x_2-x);
					result.at<Vec3b>(i,j).val[2] += src2.at<Vec3b>(y_1,x_1).val[2]*(y_2-y)*(x_2-x);

					result.at<Vec3b>(i,j).val[0] += src2.at<Vec3b>(y_2,x_1).val[0]*(y_2-y)*(x-x_1);	
					result.at<Vec3b>(i,j).val[1] += src2.at<Vec3b>(y_2,x_1).val[1]*(y_2-y)*(x-x_1);
					result.at<Vec3b>(i,j).val[2] += src2.at<Vec3b>(y_2,x_1).val[2]*(y_2-y)*(x-x_1);
					
					result.at<Vec3b>(i,j).val[0] += src2.at<Vec3b>(y_1,x_2).val[0]*(y-y_1)*(x_2-x);
					result.at<Vec3b>(i,j).val[1] += src2.at<Vec3b>(y_1,x_2).val[1]*(y-y_1)*(x_2-x);
					result.at<Vec3b>(i,j).val[2] += src2.at<Vec3b>(y_1,x_2).val[2]*(y-y_1)*(x_2-x);
					
					result.at<Vec3b>(i,j).val[0] += src2.at<Vec3b>(y_2,x_2).val[0]*(y-y_1)*(x-x_1);
					result.at<Vec3b>(i,j).val[1] += src2.at<Vec3b>(y_2,x_2).val[1]*(y-y_1)*(x-x_1);
					result.at<Vec3b>(i,j).val[2] += src2.at<Vec3b>(y_2,x_2).val[2]*(y-y_1)*(x-x_1);
				}
			}	
		}
	}

	Mat fixMask(src1.rows,src1.cols,CV_8UC3,Scalar::all(0));
	Mat fixPixel(src1.rows,src1.cols,CV_8UC2,Scalar::all(0));
	
	for(int y = 0; y < src1.rows; y++)
	{
		for(int x = 0; x < src1.cols; x++)
		{	
			// if(src1.at<Vec3b>(y,x).val[0]==0 && src1.at<Vec3b>(y,x).val[1]==0 && src1.at<Vec3b>(y,x).val[2]==0)
			// {
			// 	if(result.at<Vec3b>(y,x).val[0]!=0 && result.at<Vec3b>(y,x).val[1]!=0 && result.at<Vec3b>(y,x).val[2]!=0)
			// 	{
			// 		fixMask.at<Vec3b>(y,x) = result.at<Vec3b>(y,x);
			// 	}
			// }

			//kind of blending, stitch better, but have ghost
			double result_distance = result.at<Vec3b>(y,x).val[2]*0.299 + result.at<Vec3b>(y,x).val[1]*0.587 + result.at<Vec3b>(y,x).val[0]*0.114; 
			double src1_distance = src1.at<Vec3b>(y,x).val[2]*0.299 + src1.at<Vec3b>(y,x).val[1]*0.587 + src1.at<Vec3b>(y,x).val[0]*0.114; 
			if(result_distance > src1_distance)
			{
				fixMask.at<Vec3b>(y,x) = result.at<Vec3b>(y,x);
			}
		}
	}

	for(int y = 0; y < src1.rows; y++)
	{
		for(int x = 0; x < src1.cols; x++)
		{	
			result.at<Vec3b>(y,x) = src1.at<Vec3b>(y,x);
		}
	}	

	for(int y = 0; y < src1.rows; y++)
	{
		for(int x = 0; x < src1.cols; x++)
		{	
			if(fixMask.at<Vec3b>(y,x).val[0]!=0 && fixMask.at<Vec3b>(y,x).val[1]!=0 && fixMask.at<Vec3b>(y,x).val[2]!=0)
				result.at<Vec3b>(y,x) = fixMask.at<Vec3b>(y,x);
		}
	}

	dst = result;
}

void detectOutliers(const int offset, const Feat &feat1, const Feat &feat2, const int width, const vector<array<int,2>> &matchs, vector<array<int,2>> &puredMatchs)
{
	vector<tuple<int,int>> score;
	vector<tuple<int,int>> moveVector;
	for(int i = 0; i < matchs.size(); i++)
	{	
		int x1 = get<1>(feat1.keypoints[matchs[i][0]])+offset;
		int y1 = get<0>(feat1.keypoints[matchs[i][0]]);
		int x2 = get<1>(feat2.keypoints[matchs[i][1]]);
		int y2 = get<0>(feat2.keypoints[matchs[i][1]]);
		moveVector.push_back(make_tuple(x1-x2,y1-y2));
	}

	for(int i = 0; i < matchs.size(); i++)
	{	
		int tmp = 0;
		for(int j = 0; j < matchs.size(); j++)
		{
			tmp += abs(get<0>(moveVector[i])-get<0>(moveVector[j]));
			tmp += abs(get<1>(moveVector[i])-get<1>(moveVector[j]));
		}
		//cout << tmp << endl;
		score.push_back(make_tuple(i,tmp));
	}
	// cout << "sort" << endl;
		
	sort(begin(score), end(score),[](tuple<int, int> const &t1, tuple<int, int> const &t2) {
        return get<1>(t1) < get<1>(t2);
    });

	cout << "match size: " << matchs.size() << endl;

	// 0.05 parrington best
    for(int i = 0; i < matchs.size()*0.05; i++)
    {
		//cout << get<1>(score[i]) << endl;	
		puredMatchs.push_back(matchs[get<0>(score[i])]);	
	}
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

void cylindricalWarping(const Mat &src, Mat &dst, float f)
{
	Mat result(src.rows, src.cols, src.type(), Scalar::all(0));
	int xc = src.cols/2;
	int yc = src.rows/2;
	for(int y = 0; y < src.rows; y++)
		for(int x = 0; x < src.cols; x++)
		{
			int x_ = x - xc + 1;
			int y_ = y - yc + 1;
			y_ = y_ * sqrt(1+ pow(tan(x_/f),2));
			x_ = f*tan(x_/f);
			x_ += xc - 1;
			y_ += yc - 1;
			if(x_ >= 0.0 && x_ < src.cols && y_ >= 0.0 && y_ < src.rows)
				result.at<Vec3b>(y, x) = src.at<Vec3b>(y_, x_);
		}
	dst = result;
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
	feat.num = 0;
	for(int i = 9; i < img.rows-9; i++)
		for(int j = 9; j < img.cols-9; j++)
		{
			if(R.at<double>(i,j) > R.at<double>(i-1,j) &&
			   R.at<double>(i,j) > R.at<double>(i+1,j) &&
			   R.at<double>(i,j) > R.at<double>(i,j-1) &&
			   R.at<double>(i,j) > R.at<double>(i,j+1) &&
			   R.at<double>(i,j) > 100000)
			{
				feat.keypoints.push_back(make_tuple(i,j));
				feat.num++;
			}
		}

	featureDescriptor(feat.keypoints, Io, feat.descriptors);
}

double cosineSimilarity(const tuple<int, int> v1, const tuple<int, int> v2)
{
	double sum = get<0>(v1) * get<0>(v2) + get<1>(v1) * get<1>(v2);
	double len1 = get<0>(v1) * get<0>(v1) + get<1>(v1) * get<1>(v1);
	double len2 = get<0>(v2) * get<0>(v2) + get<1>(v2) * get<1>(v2);
	len1 = sqrt(len1);
	len2 = sqrt(len2);
	return (sum/(len1*len2));
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
	// cout << "feat1: " << feat1.num << endl;
	for(int i = 0; i < feat1.num; i++)
	{	
		// cout << i << " ";
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
			//cout << match[0] << " " << match[1] << " " << max_score << endl;
			matchs.push_back(match);
		}
	}
	//cout << matchs.size() << endl;
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
