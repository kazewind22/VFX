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

double PURE_THRESH;
double MATCH_THRESH;
int right_x;
int left_x;


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
double cosineSimilarity(const tuple<int, int> v1, const tuple<int, int> v2);
void featureMatching(const Feat &feat1, const Feat &feat2, vector<array<int,2>> &matches);
void combine2Images(const Mat &src1, const Mat &src2, Mat &dst);
void detectOutliers(const int offset, const int width, const Feat &feat1, const Feat &feat2, const vector<array<int,2>> &matches, vector<array<int,2>> &puredMatches);
void transformation(const vector<Feat> &feats, const vector<array<int,2>> &puredMatches, Mat &_M, Mat &M, int imgIndex);
void stitchImages(const Mat &src1, const Mat &src2, const Mat &M, Mat &dst);
void refineImage(int origin_y,const Mat &src, Mat &dst);
void output(Mat &stitchedImage, char* name, char* para1, char* para2);
void output_refine(Mat &stitchedImage, char* name, char* para1, char* para2, int width);

int main(int argc, char** argv)
{
	string path = argv[1];
	MATCH_THRESH = atof(argv[2])/100;
	PURE_THRESH = atof(argv[3])/20;
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

	Mat stitchedImage;
	Mat image;
	Mat img1;
	Mat img2;
	Mat match;
	// This _M is used to record tranformation matrix 
	Mat _M(3,3,CV_64FC1,Scalar::all(0));
	_M.at<double>(0,0)=1;
	_M.at<double>(1,1)=1;
	_M.at<double>(2,2)=1;


	for(int imgIndex = images.size()-1; imgIndex >= 1; imgIndex--){
		cout << images.size()-imgIndex << "th iter" << endl;
		img1 = images[imgIndex-1].clone();
		img2 = images[imgIndex].clone();	

		for(int index = 0; index < feats[imgIndex-1].num; index++)
		{
			int i = get<0>(feats[imgIndex-1].keypoints[index]);
			int j = get<1>(feats[imgIndex-1].keypoints[index]);
			circle(img1,Point(j,i),2,Scalar(22));
		}

		for(int index = 0; index < feats[imgIndex].num; index++)
		{
			int i = get<0>(feats[imgIndex].keypoints[index]);
			int j = get<1>(feats[imgIndex].keypoints[index]);
			circle(img2,Point(j,i),2,Scalar(22));
		}
	

		cout << "feature matching" << endl;
		vector<array<int,2>> matches;
		featureMatching(feats[imgIndex-1], feats[imgIndex], matches);
		
		cout << "detect outliers" << endl;
		vector<array<int,2>> puredMatches;
		detectOutliers(img2.cols, img1.cols, feats[imgIndex-1], feats[imgIndex], matches, puredMatches);
	
		combine2Images(img2,img1,match);
		for(int i = 0; i < matches.size(); i++)
		{
			int x1 = get<1>(feats[imgIndex-1].keypoints[matches[i][0]])+img2.cols;
			int y1 = get<0>(feats[imgIndex-1].keypoints[matches[i][0]]);
			int x2 = get<1>(feats[imgIndex].keypoints[matches[i][1]]);
			int y2 = get<0>(feats[imgIndex].keypoints[matches[i][1]]);
			line(match, Point(x1, y1), Point(x2, y2), Scalar(rand()%256,rand()%256,rand()%256));
		}

		// ostringstream stringStreamM;
		// stringStreamM << argv[1] << "_output/" << imgIndex <<"_iter_" << argv[2] << argv[3] << "_match.jpg";
		// string nameMatch= stringStreamM.str();
		// imwrite(nameMatch,match);
		
		
		if(imgIndex == images.size()-1)
		{	
			Mat cylindrical;
			Mat mask;
			cylindricalWarping(images[imgIndex], cylindrical, mask, feats[imgIndex], focalLengths[imgIndex]);
			warped_imgs.push_back(cylindrical);
			masks.push_back(mask);
			image = warped_imgs[0].clone();
		}
		else
			image = stitchedImage;

		Mat cylindrical;
		Mat mask;
		cylindricalWarping(images[imgIndex-1], cylindrical, mask, feats[imgIndex-1], focalLengths[imgIndex-1]);
		warped_imgs.push_back(cylindrical);
		masks.push_back(mask);

		img1 = warped_imgs[images.size()-1-(imgIndex-1)].clone();
		img2 = warped_imgs[images.size()-1-imgIndex].clone();	
		
		for(int index = 0; index < feats[imgIndex-1].num; index++)
		{
			int i = get<0>(feats[imgIndex-1].keypoints[index]);
			int j = get<1>(feats[imgIndex-1].keypoints[index]);
			// circle(img1,Point(j,i),2,Scalar::all(22));
		}

		for(int index = 0; index < feats[imgIndex].num; index++)
		{
			int i = get<0>(feats[imgIndex].keypoints[index]);
			int j = get<1>(feats[imgIndex].keypoints[index]);
			// circle(img2,Point(j,i),2,Scalar::all(22));
		}
	

		Mat match_warp;
		combine2Images(img2,img1,match_warp);
		for(int i = 0; i < puredMatches.size(); i++)
		{
			int x1 = get<1>(feats[imgIndex-1].keypoints[puredMatches[i][0]])+img2.cols;
			int y1 = get<0>(feats[imgIndex-1].keypoints[puredMatches[i][0]]);
			int x2 = get<1>(feats[imgIndex].keypoints[puredMatches[i][1]]);
			int y2 = get<0>(feats[imgIndex].keypoints[puredMatches[i][1]]);
			line(match_warp, Point(x1, y1), Point(x2, y2), Scalar(rand()%256,rand()%256,rand()%256));
		}

		// ostringstream ss;
		// ss << argv[1] << "_output/" << imgIndex <<"_iter_" << argv[2] << argv[3] << "_warped_match.jpg";
		// string nameWarpedMatch = ss.str();
		// imwrite(nameWarpedMatch,match_warp);

		Mat M;
		cout << "transforamtion matrix" << endl;
		transformation(feats, puredMatches, _M, M, imgIndex);
		
		cout << "image stitching" << endl;
		stitchImages(image,img1,M,stitchedImage);
	
		// ostringstream sss;
		// sss << argv[1] << "_output/" << argv[1] << imgIndex << "_stitched.jpg";
		// string stitchName= sss.str();
		// imwrite(stitchName,stitchedImage);

	}
	output(stitchedImage, argv[1], argv[2], argv[3]);
	output_refine(stitchedImage, argv[1], argv[2], argv[3], images[0].rows);
	// imshow("match", stitchedImage);
	// waitKey(0);
	return 0;
}

void output_refine(Mat &stitchedImage, char* name, char* para1, char* para2, int width)
{
	Mat refinedImage;
	refineImage(width,stitchedImage,refinedImage);
	cout << "stitch size: " << refinedImage.size() << endl;
	ostringstream s;
	s << name << "_output/" << name << para1 << para2 << "_refined.jpg";
	string stitchName= s.str();
	imwrite(stitchName,refinedImage);

}

void output(Mat &stitchedImage, char* name, char* para1, char* para2)
{
	cout << "refined stitch size: " << stitchedImage.size() << endl;
	ostringstream s;
	s << name << "_output/" << name << para1 << para2 << "_stitched.jpg";
	string stitchName= s.str();
	imwrite(stitchName,stitchedImage);

}

void transformation(const vector<Feat> &feats, const vector<array<int,2>> &puredMatches, Mat &_M, Mat &M, int imgIndex)
{
	vector<Point2f> obj;
	vector<Point2f> scene;
	for( int i = 0; i < puredMatches.size(); i++ )
	{	
		Point2f a(get<1>(feats[imgIndex-1].keypoints[puredMatches[i][0]]), get<0>(feats[imgIndex-1].keypoints[puredMatches[i][0]]));
		Point2f b(get<1>(feats[imgIndex].keypoints[puredMatches[i][1]]), get<0>(feats[imgIndex].keypoints[puredMatches[i][1]]));
		//-- Get the keypoints from the good matches
		obj.push_back(b);
		scene.push_back(a);
	}

	Mat objVector = Mat(puredMatches.size(),3,CV_64F,Scalar::all(0));
	Mat sceneVector = Mat(puredMatches.size(),3,CV_64F,Scalar::all(0));
	for(int i = 0; i < puredMatches.size(); i++)
	{
		objVector.at<double>(i,0) = obj[i].x;
		objVector.at<double>(i,1) = obj[i].y;
		objVector.at<double>(i,2) = 1;
		sceneVector.at<double>(i,0) = scene[i].x;
		sceneVector.at<double>(i,1) = scene[i].y;
		sceneVector.at<double>(i,2) = 1;
	}
	
	Mat tmpM;
	// = findHomography(sceneVector,objVector,CV_RANSAC);
	solve(sceneVector, objVector, tmpM, DECOMP_NORMAL );
	tmpM = tmpM.t();
	M = _M*tmpM;
	_M = M;
}


void refineImage(int origin_y, const Mat &src, Mat &dst)
{
	// double m = (double)(src.rows - origin_y)/(double)src.cols;
	double m = 0;	
	// cout << "src rows: "<< src.rows << "origin_y: " << origin_y << "m = " << m << endl;
	Mat result(origin_y*0.9, src.cols*0.95, CV_8UC3, Scalar::all(0));
	for (int x = 0; x < result.cols; x++)
	{	
		int drift = x*m;
		for (int y = 0; y < result.rows; y++)
		{	
			int _x = x + 0.01 * src.cols;
			int _y = y + drift + 0.05 * origin_y;
			if(_y >= 0 && _y < src.rows && _x >=0 && _x < src.cols)
				result.at<Vec3b>(y,x) = src.at<Vec3b>(_y,_x);
		}	
	}
	dst = result;
}

void stitchImages(const Mat &src1, const Mat &src2,const Mat &M, Mat &dst)
{

	// result size
	Mat T = M.inv(DECOMP_LU);
	Mat scr2Index = Mat(src2.rows,src2.cols,CV_32FC2,Scalar::all(0));
	double max_x = 0; 
	double max_y = 0;
	double min_x = 1000000; 
	double min_y = 1000000;
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
		left_x = min_x;
	}

	right_x = src1.cols;

	Mat result(max(src1.rows, (int)max_y)+1, (int)max_x+1, CV_8UC3,Scalar::all(0));

	for(int i = 0; i < result.rows; i++)
	{
		for(int j = 0; j < result.cols; j++)
		{	
			//Forward warping not good, we use inverse warping.
			double y = (T.at<double>(1,0)*j+T.at<double>(1,1)*i+T.at<double>(1,2));
			double x = (T.at<double>(0,0)*j+T.at<double>(0,1)*i+T.at<double>(0,2));
			if(y >= 0 && y < src2.rows && x >= 0 && x < src2.cols)
			{
				double y_1 = floor(y);
				double x_1 = floor(x);
				double y_2 = y_1+1;
				double x_2 = x_1+1;
				if(y_2>=src2.rows)
					y_2--;
				if(y_2>=src2.rows)
					y_2--;
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
			if(x>900)
			{
				if(src1.at<Vec3b>(y,x).val[0]!=0 && src1.at<Vec3b>(y,x).val[1]!=0 && src1.at<Vec3b>(y,x).val[2]!=0)
				{
					if(result.at<Vec3b>(y,x).val[0]!=0 && result.at<Vec3b>(y,x).val[1]!=0 && result.at<Vec3b>(y,x).val[2]!=0)
					{
						fixMask.at<Vec3b>(y,x).val[0] += result.at<Vec3b>(y,x).val[0]*(x-left_x)/(right_x-left_x);
						fixMask.at<Vec3b>(y,x).val[0] += src1.at<Vec3b>(y,x).val[0]*(right_x-x)/(right_x-left_x);
						fixMask.at<Vec3b>(y,x).val[1] += result.at<Vec3b>(y,x).val[1]*(x-left_x)/(right_x-left_x);
						fixMask.at<Vec3b>(y,x).val[1] += src1.at<Vec3b>(y,x).val[1]*(right_x-x)/(right_x-left_x);
						fixMask.at<Vec3b>(y,x).val[2] += result.at<Vec3b>(y,x).val[2]*(x-left_x)/(right_x-left_x);
						fixMask.at<Vec3b>(y,x).val[2] += src1.at<Vec3b>(y,x).val[2]*(right_x-x)/(right_x-left_x);
						// if(right_x - x > x - left_x)
						// 	fixMask.at<Vec3b>(y,x) = src1.at<Vec3b>(y,x);
						// else
						// 	fixMask.at<Vec3b>(y,x) = result.at<Vec3b>(y,x);
					}
				}
				else
				{
					fixMask.at<Vec3b>(y,x) = result.at<Vec3b>(y,x);
				}
			}
			else
			{
				if(right_x - x > x - left_x)
					fixMask.at<Vec3b>(y,x) = src1.at<Vec3b>(y,x);
				else
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

void detectOutliers(const int offset, const int width, const Feat &feat1, const Feat &feat2, const vector<array<int,2>> &matches, vector<array<int,2>> &puredMatches)
{
	vector<tuple<int,int>> score;
	vector<tuple<int,int>> moveVector;
	for(int i = 0; i < matches.size(); i++)
	{	
		int x1 = get<1>(feat1.keypoints[matches[i][0]])+offset;
		int y1 = get<0>(feat1.keypoints[matches[i][0]]);
		int x2 = get<1>(feat2.keypoints[matches[i][1]]);
		int y2 = get<0>(feat2.keypoints[matches[i][1]]);
		moveVector.push_back(make_tuple(x1-x2,y1-y2));
	}

	// int n = 3;
	// double p = 0.5;
	// int k = 35;
	// vector<tuple<double,double,int>> ransacResult;
	
	// for(int i = 0; i < k; i++)
	// {
	// 	int count = 0;
	// 	double x = 0;
	// 	double y = 0;
	// 	for(int j = 0; j < n; j++)
	// 	{
	// 		int index = rand() % matches.size();
	// 		x += get<0>(moveVector[index]);
	// 		y += get<1>(moveVector[index]);
	// 	}
	// 	y = y/n;
	// 	x = x/n;
	// 	for(int j = 0; j < matches.size(); j++)
	// 	{
	// 		if(cosineSimilarity(make_tuple(x,y),moveVector[j]) > 0.85)
	// 		{
	// 			count++;
	// 		}
	// 	}
	// 	ransacResult.push_back(make_tuple(x,y,count));
	// }

	// sort(ransacResult.begin(), ransacResult.end(),
	// 	[](tuple<double, double ,int> const &t1, tuple<double, double, int> const &t2)
	// 	{
	// 		return get<2>(t1) > get<2>(t2);
	// 	});

	// for (int i = 0; i < matches.size(); i++)
	// {
	// 	if(cosineSimilarity(make_tuple(get<0>(ransacResult[0]),get<1>(ransacResult[0])),moveVector[i]) > 0.85)
	// 	{
	// 		puredMatches.push_back(matches[i]);
	// 	}
	// }

	for(int i = 0; i < matches.size(); i++)
	{	
		int tmp = 0;
		for(int j = 0; j < matches.size(); j++)
		{	
			int tmp_a = 0;
			int tmp_b = 0;
			tmp_a = abs(get<0>(moveVector[i])-get<0>(moveVector[j])) * abs(get<0>(moveVector[i])-get<0>(moveVector[j]));
			tmp_b = abs(get<1>(moveVector[i])-get<1>(moveVector[j])) * abs(get<1>(moveVector[i])-get<1>(moveVector[j])); 
			tmp = (int)sqrt(tmp_a+tmp_b);
		}
		//cout << tmp << endl;
		score.push_back(make_tuple(i,tmp));
	}
		
	sort(begin(score), end(score),[](tuple<int, int> const &t1, tuple<int, int> const &t2) {
        return get<1>(t1) < get<1>(t2);
    });

	// 0.05 parrington best
    for(int i = 0; i < matches.size()*PURE_THRESH; i++)
    {
		//cout << get<1>(score[i]) << endl;	
		puredMatches.push_back(matches[get<0>(score[i])]);	
	}
	cout << "pured matching: " << puredMatches.size() << endl;		
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
 					if(x+i < 0 || x+i >= src.cols)
 						continue;
 					for(int j = -2; j <= 2; j++)
 					{
 						if(y+j < 0 || y+j >= src.rows)
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
	vector<tuple<int, int, int>> scores;
	for(int i = 9; i < img.rows-9; i++)
		for(int j = 9; j < img.cols-9; j++)
		{
			if(R.at<double>(i,j) > R.at<double>(i-1,j) &&
			   R.at<double>(i,j) > R.at<double>(i+1,j) &&
			   R.at<double>(i,j) > R.at<double>(i,j-1) &&
			   R.at<double>(i,j) > R.at<double>(i,j+1) &&
			   R.at<double>(i,j) > THRESH)
			{
				scores.push_back(make_tuple(R.at<double>(i,j), i, j));
			}
		}
	sort(scores.begin(), scores.end(),
		[](tuple<double, int ,int> const &t1, tuple<double, int, int> const &t2)
		{
			return get<0>(t1) > get<0>(t2);
		});
	for(int i = 0;i < scores.size(); i++)
	{
		feat.keypoints.push_back(make_tuple(get<1>(scores[i]), get<2>(scores[i])));
		feat.num++;
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

void featureMatching(const Feat &feat1, const Feat &feat2, vector<array<int,2>> &matches)
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
		if(max_score > MATCH_THRESH)
		{
			array<int,2> match = {i, max_index};
			//cout << match[0] << " " << match[1] << " " << max_score << endl;
			matches.push_back(match);
		}
	}
	cout << "matching size: " << matches.size() << endl;
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
