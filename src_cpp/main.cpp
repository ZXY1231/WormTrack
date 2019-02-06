#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

#include "WienerFilter.h"

int main(void)
{
	string testImage = "C.elegans_203550_0033.tif";
	Mat src = imread(testImage, CV_16UC1);
	Mat out;


	// Call to WienerFilter function with a 3x3 kernel and estimated noise variances
	WienerFilter(src,out, Size(3, 3));

        imwrite("C.elegans_203550_0033_cpp_winner_3_3.tif",src);
       //定义核
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(50, 50)); 
       //进行形态学操作
        morphologyEx(out,out, MORPH_TOPHAT, element);
       //显示效果图 
        imwrite("C.elegans_203550_0033_cpp_winner_3_3_rolling_50_50.tif",out);
        return 0; 
}
