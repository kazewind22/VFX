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

typedef struct feat
{
	int num = 0;
	vector<tuple<int,int>> keypoints;
	vector<array<int,128>> descriptors;
} Feat;

typedef struct myparam
{
	int FEATURE_THRESH;
	double MATCH_THRESH;
	double PURE_THRESH;
	bool do_stitching;
	bool do_cropping;
	bool do_rectangling;
	string photo_path;
	string panorama_path;
} MyParam;

void loadImageSeq(string path, vector<Mat> &images, vector<float> &focalLengths);

void cylindricalWarping(const Mat &src, Mat &dst, Mat &mask, Feat &feat, float f);
void gradI(const Mat &src, Mat &Ix, Mat &Iy, Mat &Io);
double ResponseFunction(const Mat &M, const double k);
void featureDescriptor(const vector<tuple<int, int>> &keypoints, const Mat &Io, vector<array<int,128>> &descriptors);
void getFeatures(const Mat &img, Feat &feat, const MyParam param);
double cosineSimilarity(const array<int, 128> des1, const array<int, 128> des2);
double cosineSimilarity(const tuple<int, int> v1, const tuple<int, int> v2);
void featureMatching(const Feat &feat1, const Feat &feat2, vector<array<int,2>> &matches, const MyParam param);
void combine2Images(const Mat &src1, const Mat &src2, Mat &dst);
void detectOutliers(const int offset, const int width, const Feat &feat1, const Feat &feat2, const vector<array<int,2>> &matches, vector<array<int,2>> &puredMatches, const MyParam param);
void transformation(const vector<Feat> &feats, const vector<array<int,2>> &puredMatches, Mat &_M, Mat &M, int imgIndex);
void stitchImages(const Mat &src1, const Mat &src2, const Mat &M, Mat &dst);
void refineImage(int origin_y,const Mat &src, Mat &dst);
void output(Mat &stitchedImage, char* name, char* para1, char* para2);
void output_refine(Mat &stitchedImage, char* name, char* para1, char* para2, int width);

void preprocess(const Mat &src, Mat &dst, Mat &mask);
Mat localWarping(const Mat &img, const Mat &mask, vector<array<int,2>> &meshVertexs, int &meshx, int &meshy);
bool findSubimage(const Mat &img, const Mat &mask, Rect &sub);
void computeFullEnergy(Mat &img, Mat &energy);
void computeEnergyAfterSeamRemoval(const Mat &image, Mat& energy, vector<uint> seam);
vector<uint> findVerticalSeam(const Mat& image, const Mat &energy);
vector<uint> findHorizontalSeam(const Mat &img, const Mat &energy);
void completeVerticalSeam(Mat& image, Mat& mask, Mat& displace,  const vector<uint> &seam, bool left);
void completeHorizontalSeam(Mat& image, Mat& mask, Mat& displace,  const vector<uint> &seam, bool up);
void showVerticalSeam(Mat &img, const vector<uint> seam);
void showHorizontalSeam(Mat &img, const vector<uint> seam);
