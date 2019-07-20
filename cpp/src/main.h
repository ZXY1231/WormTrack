#include <opencv2/core.hpp> 
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/highgui.hpp> 
#include <opencv2/tracking.hpp>

using namespace cv;

void help();
void tracker_pos_update(Mat *whole_img, Mat *body_new, unsigned int x_ori_old,
        unsigned int y_ori_old, unsigned int x_ori_new, unsigned y_ori_new);
