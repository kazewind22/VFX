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
#include "panorama.h"

using namespace cv;
using namespace std;

bool is_numerical(char *str)
{
	int c = 0;
	while(*str != '\0')
	{
		if(isdigit(*str))
			c++;
		str++;
	}
	return c > 0;
}

string train_help()
{
	return string(
			"usage: panorama [parameters] photo_dir\n"
			"\n"
			"parameters:\n"
			"-f <feature_thresh>: set the threshold for feature detection (default 100000)\n"
			"-m <match_thresh>: set the threshold for feature matching (default 0.8)\n"
			"-p <pure_thresh>: set the threshold for puring features (default 0.85)\n"
			"-o <panorama_path>: set the path to panorama image\n"
			"--no_stitching: turn off stitching process, crop or rectangling specific panoramaat <panorama_path>\n"
			"--do_cropping: turn on cropping to refined the panorama\n"
			"--do_rectangling: turn on rectangling technique to refine the panorama\n"
			);
}

MyParam parse_param(int argc, char **argv)
{
	vector<string> args;
	for(int i = 0; i < argc; i++)
		args.push_back(string(argv[i]));

	if(argc == 1)
		throw invalid_argument(train_help());

	MyParam param;
	param.FEATURE_THRESH = 100000;
	param.MATCH_THRESH = 0.85;
	param.PURE_THRESH = 0.8;
	param.do_stitching = true;
	param.do_cropping = false;
	param.do_rectangling = false;

	int i = 0;
	for(i = 1; i < argc; i++)
	{
		if(args[i].compare("-f") == 0)
		{
			if((i+1) >= argc)
				throw invalid_argument("need to specify threshold for feature detction");
			i++;

			if(!is_numerical(argv[i]))
				throw invalid_argument("-f should be followed by a interger");
			param.FEATURE_THRESH = atoi(argv[i]);
		}
		else if(args[i].compare("-m") == 0)
		{
			if((i+1) >= argc)
				throw invalid_argument("need to specify threshold for feature matching");
			i++;

			if(!is_numerical(argv[i]))
				throw invalid_argument("-m should be followed by a number");
			param.MATCH_THRESH = atof(argv[i]);
		}
		else if(args[i].compare("-p") == 0)
		{
			if((i+1) >= argc)
				throw invalid_argument("need to specify threshold for puring features");
			i++;

			if(!is_numerical(argv[i]))
				throw invalid_argument("-p should be followed by a number");
			param.PURE_THRESH = atof(argv[i]);
		}
		else if(args[i].compare("--no_stitching") == 0)
		{
			param.do_stitching = false;

		}
		else if(args[i].compare("--do_cropping") == 0)
		{
			param.do_cropping = true;
		}
		else if(args[i].compare("--do_rectangling") == 0)
		{
			param.do_rectangling = true;
		}
		else if(args[i].compare("-o") == 0)
		{
			if((i+1) >= argc)
				throw invalid_argument("need to specify path after -o");
			i++;

			param.panorama_path = string(args[i]);
		}
		else
		{
			break;
		}
	}

	if(i >= argc)
		throw invalid_argument("photo directory not specified");
	param.photo_path = string(args[i++]);
	if(param.panorama_path.empty())
		param.panorama_path = param.photo_path+string("/pano.jpg");

	return param;
}

int main(int argc, char** argv)
{
	MyParam param;
	try
	{
		param = parse_param(argc, argv);
	}
	catch(invalid_argument &e)
	{
		cerr << e.what() << endl;
		return 1;
	}

	vector<Mat> images;
	vector<float> focalLengths;
	vector<Mat> warped_imgs;
	vector<Mat> masks;
	vector<Feat> feats;

	Mat stitchedImage;
	Mat image;
	Mat img1;
	Mat img2;

	loadImageSeq(param.photo_path, images, focalLengths);

	if(param.do_stitching)
	{
		for(int i = 0; i < images.size(); i++)
		{
			Feat feat;
			getFeatures(images[i], feat, param);
			feats.push_back(feat);
		}

		// This _M is used to record tranformation matrix
		Mat _M(3,3,CV_64FC1,Scalar::all(0));
		_M.at<double>(0,0)=1;
		_M.at<double>(1,1)=1;
		_M.at<double>(2,2)=1;

		for(int imgIndex = images.size()-1; imgIndex >= 1; imgIndex--)
		{
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
			featureMatching(feats[imgIndex-1], feats[imgIndex], matches, param);

			cout << "detect outliers" << endl;
			vector<array<int,2>> puredMatches;
			detectOutliers(img2.cols, img1.cols, feats[imgIndex-1], feats[imgIndex], matches, puredMatches, param);

			if(imgIndex == images.size()-1)
			{
				Mat cylindrical;
				Mat mask;
				cylindricalWarping(images[imgIndex], cylindrical, mask, feats[imgIndex], focalLengths[imgIndex]);
				warped_imgs.push_back(cylindrical.clone());
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

			Mat M;
			cout << "transforamtion matrix" << endl;
			transformation(feats, puredMatches, _M, M, imgIndex);

			cout << "image stitching" << endl;
			stitchImages(image,img1,M,stitchedImage);
		}
		imwrite(param.panorama_path, stitchedImage);
	}
	else
	{
		cout << "reading image: " << param.panorama_path << endl;
		stitchedImage = imread(param.panorama_path);
	}
	if(param.do_cropping)
	{
		Mat refinedImage;
		size_t offset = param.panorama_path.rfind('/');
		string refined_path = param.panorama_path;
		refined_path.insert(offset+1, "cropped_");
		cout << "cropping image: " << param.panorama_path << endl;
		refineImage(images[0].rows, stitchedImage, refinedImage);
		imwrite(refined_path, refinedImage);
		cout << "Cropping Done!" << endl;
	}
	if(param.do_rectangling)
	{
		Mat _stitchedImage;
		Mat mask;
		vector<Point2f> meshVertexs;

		string rectangled_path = param.panorama_path;
		size_t offset = param.panorama_path.rfind('/');
		rectangled_path.insert(offset+1, "rectangled_");
		cout << "rectangling image: " << param.panorama_path << endl;

		cout << "Local warping stage:" << endl;
		cout << "start preprocessing" << endl;
		preprocess(stitchedImage, _stitchedImage, mask);
		cout << "finish preprocessing" << endl;
		int meshx, meshy;
		cout << "start local warping" << endl;
		localWarping(_stitchedImage, mask, meshVertexs, meshx, meshy);
		cout << "finish local warping" << endl;
		cout << "start global warping" << endl;
		Mat rectangledImage;
		globalWarping(_stitchedImage, rectangledImage, meshVertexs);
		cout << "finish global warping" << endl;
		imwrite(rectangled_path, rectangledImage);
		cout << "Rectangling Done!" << endl;
	}

	return 0;
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
	};

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
	int right_x = src1.cols;
	int left_x;
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

			// double result_distance = result.at<Vec3b>(y,x).val[2]*0.299 + result.at<Vec3b>(y,x).val[1]*0.587 + result.at<Vec3b>(y,x).val[0]*0.114;
			// double src1_distance = src1.at<Vec3b>(y,x).val[2]*0.299 + src1.at<Vec3b>(y,x).val[1]*0.587 + src1.at<Vec3b>(y,x).val[0]*0.114;
			// if(result_distance > src1_distance)
			// {
			// 	fixMask.at<Vec3b>(y,x) = result.at<Vec3b>(y,x);
			// }
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

void detectOutliers(const int offset, const int width, const Feat &feat1, const Feat &feat2, const vector<array<int,2>> &matches, vector<array<int,2>> &puredMatches, const MyParam param)
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
	for(int i = 0; i < matches.size()*param.PURE_THRESH; i++)
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

void getFeatures(const Mat &img, Feat &feat, const MyParam param)
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
					R.at<double>(i,j) > param.FEATURE_THRESH)
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

void featureMatching(const Feat &feat1, const Feat &feat2, vector<array<int,2>> &matches, const MyParam param)
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
		if(max_score > param.MATCH_THRESH)
		{
			array<int,2> match = {i, max_index};
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

void preprocess(const Mat &src, Mat &dst, Mat &mask)
{
	mask = Mat(src.rows, src.cols, CV_8UC1, Scalar::all(128));
	Mat img = src.clone();
	vector<array<int,2>> queue;
	for(int i = 0; i < img.rows; i++)
	{
		mask.at<uchar>(i,0) = 0;
		mask.at<uchar>(i,img.cols-1) = 0;
	}
	for(int j = 0; j < img.cols; j++)
	{
		mask.at<uchar>(0,j) = 0;
		mask.at<uchar>(img.rows-1,j) = 0;
	}
	for(int i = 1; i < img.rows-1; i++)
	{
		array<int,2> tmp = {i,1};
		queue.push_back(tmp);
		tmp[1] = img.cols-2;
		queue.push_back(tmp);
	}
	for(int j = 1; j < img.cols-1; j++)
	{
		array<int,2> tmp = {1,j};
		queue.push_back(tmp);
		tmp[0] = img.rows-2;
		queue.push_back(tmp);
	}
	while(queue.size()>0)
	{
		array<int,2> point = queue.front();
		queue.erase(queue.begin());
		int i = point[0], j = point[1];
		if(mask.at<uchar>(i,j)!=128)
			continue;
		if((img.at<Vec3b>(i,j).val[0] <= 15 &&
			img.at<Vec3b>(i,j).val[1] <= 15 &&
			img.at<Vec3b>(i,j).val[2] <= 15))
		{
			mask.at<uchar>(i,j) = 1;
			if(mask.at<uchar>(i-1,j)==128)
			{
				array<int,2> tmp = {i-1,j};
				queue.push_back(tmp);
			}
			if(mask.at<uchar>(i+1,j)==128)
			{
				array<int,2> tmp = {i+1,j};
				queue.push_back(tmp);
			}
			if(mask.at<uchar>(i,j-1)==128)
			{
				array<int,2> tmp = {i,j-1};
				queue.push_back(tmp);
			}
			if(mask.at<uchar>(i,j+1)==128)
			{
				array<int,2> tmp = {i,j+1};
				queue.push_back(tmp);
			}
		}
		else
		{
			mask.at<uchar>(i,j) = 255;
		}
	}
	for(int i = 0; i < mask.rows; i++)
		for(int j = 0; j < mask.cols; j++)
			if(mask.at<uchar>(i,j)!=128)
			{
				mask.at<uchar>(i,j) = 0;
				img.at<Vec3b>(i,j) = Vec3b(0,0,0);
			}
	int top = img.rows-1, bot = 0, left = img.cols-1, right = 0;
	for(int i = 0; i < mask.rows; i++)
		for(int j = 0; j < mask.cols; j++)
		{
			if(mask.at<uchar>(i,j))
			{
				top = std::min(top,i);
				bot = std::max(bot,i);
				left = std::min(left,j);
				right = std::max(right,j);
			}
		}
	Rect rect = Rect(left,top,right-left,bot-top);
	dst = img(rect);
	mask = mask(rect);
	//imwrite("crop.jpg",dst);
	//imwrite("mask.jpg",mask);
}

Mat localWarping(const Mat &img, const Mat &mask, vector<Point2f> &meshVertexs, int &meshx, int &meshy)
{
	Mat localImg = img.clone();
	Mat localMask = mask.clone();

	Mat energy = Mat(img.rows, img.cols, CV_32S, Scalar(195075));
	computeFullEnergy(localImg, energy);

	Mat displace_x = Mat(img.rows, img.cols, CV_32S, Scalar::all(0));
	Mat displace_y = Mat(img.rows, img.cols, CV_32S, Scalar::all(0));
	for(int i = 0; i < img.rows; i++)
		for(int j = 0; j < img.cols; j++)
		{
			displace_x.at<int32_t>(i,j) = j;
			displace_y.at<int32_t>(i,j) = i;
		}

	Rect sub = Rect(0,0,10,10);
	int count = 0;
	vector<uint> seam;
	vector<uint> seam_pre;

	while(((sub.br().x - sub.tl().x > 2) && (sub.br().y - sub.tl().y > 2)) && count < 500)
	{
		count++;
		bool direction = findSubimage(localImg, localMask, sub);
		Mat subImage = localImg(sub);
		Mat subEnergy = energy(sub);
		Mat subMask = localMask(sub);
		Mat subDisX = displace_x(sub);
		Mat subDisY = displace_y(sub);

		if(direction)
		{
			seam = findVerticalSeam(subImage, subEnergy);
			if(seam == seam_pre)
			{
				showVerticalSeam(subImage, seam);
			}
			completeVerticalSeam(subImage, subMask, subDisX, seam, !sub.tl().x);
			//showVerticalSeam(subImage, seam);
			computeFullEnergy(localImg, energy);
		}
		else
		{
			seam = findHorizontalSeam(subImage, subEnergy);
			if(seam == seam_pre)
			{
				showHorizontalSeam(subImage, seam);
			}
			completeHorizontalSeam(subImage, subMask, subDisY, seam, !sub.tl().y);
			//showHorizontalSeam(subImage, seam);
			computeFullEnergy(localImg, energy);
		}
		seam_pre = seam;
		//imwrite(string("./square_output/complete")+to_string(count)+string(".jpg"),localImg);
	}

	meshx = sqrt((double)localImg.cols/(double)localImg.rows)*22.0;
	meshy = sqrt((double)localImg.rows/(double)localImg.cols)*22.0;

	double x_ = (double)(localImg.cols-5)/(meshx-1);
	double y_ = (double)(localImg.rows-5)/(meshy-1);

	for(int i = 0; i < meshy; i++)
		for(int j = 0; j < meshx; j++)
	{
		int i_ = y_*i + 2;
		int j_ = x_*j + 2;
		//cout << "Displacement at " << i_ << ", " << j_ << " : "
		//	 << displace_y.at<int32_t>(i_,j_) << ", "
		//	 << displace_x.at<int32_t>(i_,j_) << endl;
		//circle(localImg,Point(j_,i_),3,Scalar(22));
		meshVertexs.push_back(Point(displace_x.at<int32_t>(i_,j_), displace_y.at<int32_t>(i_,j_)));
	}
	return localImg;
}

bool findSubimage(const Mat &img, const Mat &mask, Rect &sub)
{
	int cuts[4][2];
	int tmp[4] = {0};
	int length[4] = {0};
	for(int j = 1; j < img.cols; j++)
	{
		if(mask.at<uchar>(0,j) > 0)
		{
			if((j - tmp[0])>length[0])
			{
				cuts[0][0] = tmp[0];
				cuts[0][1] = j;
				length[0] = j - tmp[0];
			}
			tmp[0] = j;
		}

		if(mask.at<uchar>(img.rows-1,j) > 0)
		{
			if((j - tmp[1])>length[1])
			{
				cuts[1][0] = tmp[1];
				cuts[1][1] = j;
				length[1] = j - tmp[1];
			}
			tmp[1] = j;
		}

		if(j == img.cols-1)
		{
			for(int k = 0; k < 2; k++)
			{
				if((j - tmp[k])>length[k])
				{
					cuts[k][0] = tmp[k];
					cuts[k][1] = j;
					length[k] = j - tmp[k];
				}
			}
		}
	}

	for(int i = 1; i < img.rows; i++)
	{
		if(mask.at<uchar>(i,0) > 0)
		{
			if((i - tmp[2])>length[2])
			{
				cuts[2][0] = tmp[2];
				cuts[2][1] = i;
				length[2] = i - tmp[2];
			}
			tmp[2] = i;
		}

		if(mask.at<uchar>(i,img.cols-1) > 0)
		{
			if((i - tmp[3])>length[3])
			{
				cuts[3][0] = tmp[3];
				cuts[3][1] = i;
				length[3] = i - tmp[3];
			}
			tmp[3] = i;
		}

		if(i == img.rows-1)
		{
			for(int k = 2; k < 4; k++)
			{
				if((i - tmp[k])>length[k])
				{
					cuts[k][0] = tmp[k];
					cuts[k][1] = i;
					length[k] = i - tmp[k];
				}
			}
		}
	}

	int index = 0;
	for(int i = 0; i < 4; i++)
	{
		if(length[i]>length[index])
		{
			index = i;
		}
	}
	cuts[index][1]-=1;
	cuts[index][0]+=1;
	if((cuts[index][1]-cuts[index][0])<=0)
	{
		sub = Rect(0,0,0,0);
		return false;
	}
	if(index==0 || index==1)
	{
		sub = Rect(cuts[index][0],index%2,cuts[index][1]-cuts[index][0],img.rows-1);
		return false;
	}
	else
	{
		sub = Rect(index%2, cuts[index][0], img.cols-1, cuts[index][1]-cuts[index][0]);
		return true;
	}
}

void computeFullEnergy(Mat &img, Mat &energy) {
	//Ensure that the size of the energy matrix matches that of the image
	energy.create(img.rows, img.cols, CV_32S);

	//Scan through the image and update the energy values. Ignore boundary pixels.
	for (int i = 1; i < img.rows-1; ++i) {
		uchar* prev = img.ptr<uchar>(i-1);	//Pointer to previous row
		uchar* curr = img.ptr<uchar>(i);	//Pointer to current row
		uchar* next = img.ptr<uchar>(i+1);	//Pointer to next row

		for (int j = 1; j < img.cols-1; ++j) {
			int val = 0;
			//Energy al", "ong the x-axis
			val += (prev[3*j]-next[3*j]) * (prev[3*j]-next[3*j]);
			val += (prev[3*j+1]-next[3*j+1]) * (prev[3*j+1]-next[3*j+1]);
			val += (prev[3*j+2]-next[3*j+2]) * (prev[3*j+2]-next[3*j+2]);

			//Energy along the y-axis
			val += (curr[3*j+3]-curr[3*j-3]) * (curr[3*j+3]-curr[3*j-3]);
			val += (curr[3*j+4]-curr[3*j-2]) * (curr[3*j+4]-curr[3*j-2]);
			val += (curr[3*j+5]-curr[3*j-1]) * (curr[3*j+5]-curr[3*j-1]);

			energy.at<uint32_t>(i, j) = val;
		}
	}
}

void computeEnergyAfterSeamRemoval(const Mat &image, Mat& energy, vector<uint> seam) {
	Mat tmp = Mat(image.rows, image.cols, CV_32S, Scalar(195075));
	for (unsigned int row = 0; row < (uint)image.rows; ++row) {
		for (unsigned int col = 0; col < (uint)image.cols; ++col) {
			if (col < seam[row]-1)	tmp.at<uint32_t>(row, col) = energy.at<uint32_t>(row, col);
			if (col > seam[row])	tmp.at<uint32_t>(row, col) = energy.at<uint32_t>(row, col+1);
			if (col == seam[row] || col == seam[row]-1) {
				Vec3b l = image.at<Vec3b>(row, col-1);
				Vec3b r = image.at<Vec3b>(row, col+1);
				Vec3b u = image.at<Vec3b>(row-1, col);
				Vec3b d = image.at<Vec3b>(row+1, col);
				int val = (l[0]-r[0])*(l[0]-r[0]) + (l[1]-r[1])*(l[1]-r[1]) + (l[2]-r[2])*(l[2]-r[2]) +
						(u[0]-d[0])*(u[0]-d[0]) + (u[1]-d[1])*(u[1]-d[1]) + (u[2]-d[2])*(u[2]-d[2]);
				tmp.at<uint32_t>(row, col) = val;
			}
		}
	}
	energy = tmp;
}

vector<uint> findVerticalSeam(const Mat& image, const Mat &energy)
{
	vector<uint> seam(image.rows);
	unsigned int distTo[image.rows][image.cols];
	short edgeTo[image.rows][image.cols];

	//Initialize the distance and edge matrices
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			if (i == 0)
				distTo[i][j] = 0;
			else
				distTo[i][j] = numeric_limits<unsigned int>::max();
			edgeTo[i][j] = 0;
		}
	}

	// Relax the edges in topological order
	for (int row = 0; row < image.rows-1; ++row) {
		for (int col = 0; col < image.cols; ++col) {
			//Check the pixel to the bottom-left
			if (col != 0)
				if (distTo[row+1][col-1] > distTo[row][col] + energy.at<uint32_t>(row+1, col-1)) {
					distTo[row+1][col-1] = distTo[row][col] + energy.at<uint32_t>(row+1, col-1);
					edgeTo[row+1][col-1] = 1;
				}
			//Check the pixel right below
			if (distTo[row+1][col] > distTo[row][col] + energy.at<uint32_t>(row+1, col)) {
				distTo[row+1][col] = distTo[row][col] + energy.at<uint32_t>(row+1, col);
				edgeTo[row+1][col] = 0;
			}
			//Check the pixel to the bottom-right
			if (col != image.cols-1)
				if (distTo[row+1][col+1] > distTo[row][col] + energy.at<uint32_t>(row+1, col+1)) {
					distTo[row+1][col+1] = distTo[row][col] + energy.at<uint32_t>(row+1, col+1);
					edgeTo[row+1][col+1] = -1;
				}
		}
	}

	//Find the bottom of the min-path
	unsigned int min_index = 0, min = distTo[image.rows-1][0];
	for (int i = 1; i < image.cols; ++i)
		if (distTo[image.rows-1][i] < min) {
			min_index = i;
			min = distTo[image.rows-1][i];
		}

	//Retrace the min-path and update the 'seam' vector
	seam[image.rows-1] = min_index;
	for (int i = image.rows-1; i > 0; --i)
		seam[i-1] = seam[i] + edgeTo[i][seam[i]];

	return seam;
}

void completeVerticalSeam(Mat& image, Mat& mask, Mat& displace, const vector<uint> &seam, bool left)
{
	//Move all the pixels left to the seam one more pixels left
	for (int row = 0; row < image.rows; ++row) {
		if(left)
			for (int col = 0; col < seam[row]; ++col)
			{
				image.at<Vec3b>(row, col) = image.at<Vec3b>(row, col+1);
				mask.at<uchar>(row, col) = mask.at<uchar>(row, col+1);
				displace.at<int32_t>(row, col) = displace.at<int32_t>(row, col+1);
			}
		else
			for (int col = image.cols-1; col > seam[row]; --col)
			{
				image.at<Vec3b>(row, col) = image.at<Vec3b>(row, col-1);
				mask.at<uchar>(row,col) = mask.at<uchar>(row, col-1);
				displace.at<int32_t>(row, col) = displace.at<int32_t>(row, col-1);
			}
	}
}

vector<uint> findHorizontalSeam(const Mat &image, const Mat &energy)
{
	vector<uint> seam(image.cols);
	//Transpose the matrices and find the vertical seam
	Mat image_t;
	Mat energy_t;
	transpose(image, image_t);
	transpose(energy, energy_t);
	seam = findVerticalSeam(image_t, energy_t);

	return seam;
}

void completeHorizontalSeam(Mat& image, Mat& mask, Mat& displace,  const vector<uint> &seam, bool up)
{
	//Move all the pixels left to the seam one more pixels left
	for (int col = 0; col < image.cols; ++col) {
		if(up)
			for (int row = 0; row < seam[col]; ++row)
			{
				image.at<Vec3b>(row,col) = image.at<Vec3b>(row+1, col);
				mask.at<uchar>(row,col) = mask.at<uchar>(row+1, col);
				displace.at<int32_t>(row,col) = displace.at<int32_t>(row+1,col);
			}
		else
			for (int row = image.rows-1; row > seam[col]; --row)
			{
				image.at<Vec3b>(row,col) = image.at<Vec3b>(row-1, col);
				mask.at<uchar>(row,col) = mask.at<uchar>(row-1, col);
				displace.at<int32_t>(row,col) = displace.at<int32_t>(row-1,col);
			}
	}
}

void showVerticalSeam(Mat &img, const vector<uint> seam)
{
	for (int i = 0; i < img.rows; ++i)
		img.at<Vec3b>(i, seam[i]) = Vec3b(0, 0, 255);	//Set the color of the seam to Red
}

void showHorizontalSeam(Mat &img, const vector<uint> seam)
{
	for (int i = 0; i < img.cols; ++i)
		img.at<Vec3b>(seam[i], i) = Vec3b(0, 0, 255);	//Set the color of the seam to Red
}

void globalWarping(const Mat &img, Mat &dst, vector<Point2f> &Vertexs)
{
	vector<Point2f> fake;
	int meshx = sqrt((double)img.cols/(double)img.rows)*22.0;
	int meshy = sqrt((double)img.rows/(double)img.cols)*22.0;

	double x_ = (double)(img.cols-1)/(meshx-1);
	double y_ = (double)(img.rows-1)/(meshy-1);

	for(int i = 0; i < meshy; i++)
		for(int j = 0; j < meshx; j++)
		{
			int i_ = y_*i;
			int j_ = x_*j;
			fake.push_back(Point2f(j_,i_));
		}

	vector< vector<int>> tri_list;
	get_tri(img, Vertexs, tri_list);

	Mat image;
	img.convertTo(image,CV_32F);

	dst = Mat::zeros(img.size(), CV_32FC3);
	image.copyTo(dst);

	for(int i = 0; i < tri_list.size(); i++)
	{
		vector<Point2f> t_old(3), t_new(3);
		t_old[0] = Vertexs[tri_list[i][0]];
		t_old[1] = Vertexs[tri_list[i][1]];
		t_old[2] = Vertexs[tri_list[i][2]];
		t_new[0] = fake[tri_list[i][0]];
		t_new[1] = fake[tri_list[i][1]];
		t_new[2] = fake[tri_list[i][2]];
		morph_triangle(image, dst, t_old, t_new);
	}
	imwrite("final.jpg", dst);
}

int get_index(std::vector<cv::Point2f>& points, cv::Point2f pt) {
    for (int i = 0; i < points.size(); i ++) {
        if (points[i].x == pt.x && points[i].y == pt.y) {
            return i;
        }
    }
    return -1;
}


void get_delaunay_tri(cv::Subdiv2D& subdiv, cv::Rect rect, std::vector< std::vector<int> >& tri_list, std::vector<cv::Point2f>& points) {
    std::vector<cv::Vec6f> triangles;
    subdiv.getTriangleList(triangles);
    //cout << "# of triangles: " << triangles.size() << endl;

    for (int i = 0; i < triangles.size(); i ++) {
        cv::Vec6f t = triangles[i];
        cv::Point pt1 = cv::Point(t[0], t[1]);
        cv::Point pt2 = cv::Point(t[2], t[3]);
        cv::Point pt3 = cv::Point(t[4], t[5]);

        int pt1_index = get_index(points, pt1);
        int pt2_index = get_index(points, pt2);
        int pt3_index = get_index(points, pt3);

        if (pt1_index >= 0 && pt2_index >= 0 && pt3_index >= 0) {
            std::vector<int> e(3);
            e[0] = pt1_index;
            e[1] = pt2_index;
            e[2] = pt3_index;
            tri_list.push_back(e);
        }
    }
    cout << "# of triangles: " << tri_list.size() << endl;

    return;
}

void get_tri(const Mat &img, vector<Point2f> &points, vector< vector<int>> &tri_list)
{
	Rect rect(0,0,img.cols,img.rows);
	Subdiv2D subdiv(rect);

	for(int i = 0; i < points.size(); i++)
	{
		subdiv.insert(points[i]);
	}

	get_delaunay_tri(subdiv, rect, tri_list, points);
}

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src, vector<Point2f> &srcTri, vector<Point2f> &dstTri)
{
	// Given a pair of triangles, find the affine transform.
	Mat warpMat = getAffineTransform(srcTri, dstTri);

	// Apply the Affine Transform just found to the src image
	warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

void morph_triangle(Mat& img, Mat& img_warp, vector<Point2f>& old_points, vector<Point2f>& new_points) {

	// find bounding rectangle for triangle
	Rect r_old = boundingRect(old_points);
	Rect r_new = boundingRect(new_points);

	// offset points by left top corner of the respective rectangles
	vector<Point2f> old_offseted, new_offseted;
	vector<Point> new_offseted_int;

	for (int i = 0; i < 3; i++)
	{
		old_offseted.push_back(Point2f(old_points[i].x - r_old.x, old_points[i].y - r_old.y));
		new_offseted.push_back(Point2f(new_points[i].x - r_new.x, new_points[i].y - r_new.y));
		new_offseted_int.push_back(Point(new_points[i].x - r_new.x, new_points[i].y - r_new.y));
	}

	// get mask by filling triangle
	Mat mask = Mat::zeros(r_new.height, r_new.width, CV_32FC3);
	fillConvexPoly(mask, new_offseted_int, Scalar(1.0, 1.0, 1.0), 16, 0);

	// Apply warpImage to small rectangular patches
	Mat rect_img;
	img(r_old).copyTo(rect_img);

	Mat rect_warp = Mat::zeros(r_new.height, r_new.width, rect_img.type());
	applyAffineTransform(rect_warp, rect_img, old_offseted, new_offseted);

	// Copy triangular region of the rectangular patch to the output image
	multiply(rect_warp, mask, rect_warp);
	multiply(img_warp(r_new), Scalar(1.0, 1.0, 1.0) - mask, img_warp(r_new));
	img_warp(r_new) = img_warp(r_new) + rect_warp;

	//multiply(img(r_new), Scalar(1.0, 1.0, 1.0) - mask, img(r_new));
	//img(r_new) = img(r_new) + rect_warp;

}
