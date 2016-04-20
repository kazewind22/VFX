#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <cstdio>

using namespace std;
using namespace cv;

const float PI = 3.1415927;
float Lw = 4000.0;
float EPSILON = 0.01;
float PHI = 8.0;
float A = 0.2;
float ALPHA = 0.35;
float GAMMA = 1;

void createFilter(int x, int y, int height, int width, Mat &Gf, double scale){
    // set standard deviation to 1.0
    int _x,_y;
    double r = scale*scale;
    // sum is for normalization
    for (int i=0; i < height; i++){
        float* e = Gf.ptr<float>(i);
        for (int j=0; j < width; j++){
            _x = x-i;
            _y = y-j;
            e[j] = (1/(PI * r))*exp(-1 * (_x * _x + _y * _y)/r);
        }
    }
}

void findMaxArea(int height, int width, Mat &Lm, Mat &Ld, Mat &Gf){
    Mat v0 = Mat(height,width,CV_32F);
    Mat vn = Mat(height,width,CV_32F);
    Mat v = Mat(height,width,CV_32F);
    for (int i=0; i<height; i++){
        printf("%d\n",i);
        float* e = Lm.ptr<float>(i);
        float* p = Ld.ptr<float>(i);
        float* g = Gf.ptr<float>(i);
        for (int j=0; j<width; j++){
            float maxValue = 0.0;
            float value = 0.0;
            float s = 1.0;
            while(abs(maxValue)<EPSILON){
                createFilter(i,j,height,width,Gf,ALPHA*s);
                for (int k=0; k<height; k++){
                    float* _G = Gf.ptr<float>(k);
                    float* _Lm = Lm.ptr<float>(k);
                    float* _v0 = v0.ptr<float>(k);
                    for (int l=0; l<width; l++)
                        _v0[l] = _G[l]*_Lm[j];
                }
                createFilter(i,j,height,width,Gf,ALPHA*1.6*s);
                for (int k=0; k<height; k++){
                    float* _G = Gf.ptr<float>(k);
                    float* _Lm = Lm.ptr<float>(k);
                    float* _vn = vn.ptr<float>(k);
                    for (int l=0; l<width; l++)
                        _vn[l] = _G[l]*_Lm[j];
                }

               for (int k=0; k<height; k++){
                    float* _v = v.ptr<float>(k);
                    float* _v0 = v0.ptr<float>(k);
                    float* _vn = vn.ptr<float>(k);
                    for (int l=0; l<width; l++)
                        _v[l] = (_v0[l]-_vn[l])/(pow(2.0,PHI)*A/(s*s)+_v0[l]);
                }

                for (int k=0; k<height; k++){
                    float* _v = v.ptr<float>(k);
                    float* _v0 = v0.ptr<float>(k);
                    for (int l=0; l<width; l++){
                        if(_v[l]>maxValue)
                            maxValue = _v[l];
                        if(l==j)
                            value = _v0[l];
                    }
                }

                s*=1.6;
                // printf("(%d,%d): %f < %f ?\n",i,j,v,EPSILON);
                p[j] = e[j]/value;
            }
        }
    }
}



int main(int argc, char** argv) {

    if( argc != 2){
        cout <<" Usage: tonemap hdr_file_path" << endl;
        return -1;
    }

    Mat radianceMap;
    radianceMap = imread(argv[1],CV_LOAD_IMAGE_UNCHANGED);
    string filename;
    char *ptr = strtok(argv[1], "/"); // split source string with delimiter, and return the first sub-string to *ptr
    while (ptr != NULL) {
        filename = string(ptr);
        ptr = strtok(NULL, "/"); // with the first argument of strtok() being NULL,it would continue on splitting the remaining source string from previous strtok
    }

    Size s = radianceMap.size();
    int width = s.width;
    int height = s.height;

    if(!radianceMap.data){
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }

    Mat luminanceMap = Mat(height,width,CV_32F);

    cout << "height= " << height << ", width= " << width << endl;

    float minLum = 0.2126f * radianceMap.ptr<float>(0)[2] + 0.7152f * radianceMap.ptr<float>(0)[1] + 0.0722f * radianceMap.ptr<float>(0)[0];
    // Set all other values to zero
    float maxLum = 0;
    float avgLum = 0;
    float logAvgLum = 0;
    int numNotNaNPx = 0;    // number of pixels that are not NaN
    int numLogPx = 0; // number of non-zero pixels

    // For all pixels
    for (int i=0; i<height; i++){
        // Pointer to the ith row of the luminance map
        float* p = luminanceMap.ptr<float>(i);
        // Pointer to the ith row of the radiance map
        float* e = radianceMap.ptr<float>(i);
        for (int j=0; j<width; j++){
            // Compute luminance value
            p[j] = 0.1126f * e[3*j+2] + 0.7152f * e[3*j+1] + 0.1722f * e[3*j+0];
            if (p[j] > maxLum)
                maxLum = p[j];
            if (p[j] < minLum)
                minLum = p[j];
            // Compute average luminance avoiding NaN values
            if (cvIsNaN(p[j]) == 0){
                avgLum += (long double)p[j];
                numNotNaNPx++;
            }

            // Compute log average for non-zero pixels
            if (p[j] > 0){
                logAvgLum += log(p[j]);
                numLogPx++;
            }
        }
    }
    avgLum /= double(numNotNaNPx);
    logAvgLum = exp(logAvgLum/(long double)numLogPx);
    printf("alpha: %f, logAvgLum: %f\n",A, logAvgLum);
    printf("max: %f, min: %f\n",maxLum, minLum);

    Mat radianceMap8U = Mat(height,width,CV_8UC3);
    radianceMap.convertTo(radianceMap8U, CV_8UC3,255,0);

    //Caculating Lm

    Mat Lm = Mat(height,width,CV_32F);
    for (int i=0; i<height; i++){
        // Pointer to the ith row of the luminance map
        float* p = luminanceMap.ptr<float>(i);
        // Pointer to the ith row of Lm
        float* e = Lm.ptr<float>(i);
        for (int j=0; j<width; j++)
            e[j] = (A/logAvgLum)*p[j];
    }

    //Caculating Ld
    Mat Ld = Mat(height,width,CV_32F);
    // Lw value
    for (int i=0; i<height; i++){
        // Pointer to the ith row of the luminance map
        float* p = Lm.ptr<float>(i);
        // Pointer to the ith row of Lm
        float* e = Ld.ptr<float>(i);
        for (int j=0; j<width; j++)
            e[j] = p[j]*(1+p[j]/(Lw*Lw))/(1+p[j]);
    }

    Mat color = Mat(height,width,CV_32FC3);
    for (int i=0; i<height; i++){
        float* p = Ld.ptr<float>(i);
        float* e = radianceMap.ptr<float>(i);
        float* l = luminanceMap.ptr<float>(i);
        for (int j=0; j<width; j++){
            for(int c=0;c<3;c++)
                color.at<Vec3f>(i,j)[c] = e[3*j+c]*p[j]/l[j];
        }
    }

    Mat color8U = Mat(height,width,CV_8UC3);
    color.convertTo(color8U, CV_8UC3,255,0);

    Mat display = Mat(height,width,CV_32FC3);
    for (int i=0; i<height; i++){
        float* p = Ld.ptr<float>(i);
        float* e = radianceMap.ptr<float>(i);
        float* l = luminanceMap.ptr<float>(i);
        float* d = display.ptr<float>(i);
        for (int j=0; j<width; j++){
            display.at<Vec3f>(i,j)[0] = p[j]*pow(e[3*j+0]/l[j],GAMMA); //b
            display.at<Vec3f>(i,j)[1] = p[j]*pow(e[3*j+1]/l[j],GAMMA); //g
            display.at<Vec3f>(i,j)[2] = p[j]*pow(e[3*j+2]/l[j],GAMMA); //r
        }
    }

    Mat display8U = Mat(height,width,CV_8UC3);
    display.convertTo(display8U, CV_8UC3,255,0);
    // sprintf(condition,"output-%s.jpg",condition);
    char condition[50];

    sprintf(condition,"output/%s.jpg",filename.c_str());
    printf("%s\n", condition);
    imwrite(condition , display8U );

    return 0;
}



