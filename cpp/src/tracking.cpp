#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
/*
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
*/
#include "main.h"

using namespace std;

/******************************************************************************
 * Des: Track worm
 * | Version | Commit
 * | 0.1     | From python to C++ with OpenMP @ H.F. 20190404
 *
 * ***************************************************************************/
vector<String> filenames;

// Structure to store, id, pos_x, pos_y, wormbody, 
struct wormbody {
    int x = -1;
    int y = -1;
    Mat body;
    unsigned int worm_label;
    unsigned int worm_frame;
};

// Seed's information
struct seed {
    unsigned int label;
    int x0;
    int y0;
    int w;
    int h;
};

vector<vector<wormbody *> *> worms;
vector<seed *> seeds;
unsigned int worm_num = 0;
unsigned int frame_num = 0;
unsigned int window = 35; 

void tracker_init(Mat img0, string AUTO){
    // Fetch origin
    if ( AUTO.length() > 0 ){ // AUTO model: load initial position from file
        ifstream seedfile(AUTO);
        if ( !seedfile ){
            cout << "Fail to load position file" << endl;
        }else{
            worm_num = 0;
            frame_num = 0;
            seed worm_x;
            //TODO: Need to handle dirty data
            while (seedfile >> worm_x.label) {
                seedfile >> worm_x.x0 >> worm_x.y0;
                seedfile >> worm_x.w >> worm_x.h;
                seeds.push_back(&worm_x);
                ++worm_num;
                ++frame_num;
            }
        }
    }else{ // select one worm by hand
        Rect2d r = selectROI(img0);
        cout << r.x <<" " << r.y <<" "<< r.width <<" "<< r.height  << endl;
        seed worm0;
        worm0.label = 0;
        worm0.x0 = r.x;
        worm0.y0 = r.y;
        worm0.w = r.width;
        worm0.h = r.height;
        seeds.push_back( &worm0 );
        worm_num = 1;
        frame_num = 1;
    }

    // Generate first frame
    for (unsigned int i = 0; i< worm_num; ++i ){
        cout << seeds[i]->x0 << " "<< seeds[i]->y0 << endl;
        Mat g1, g2;
        Mat body;
        Mat img0_roi = img0( Rect(seeds[i]->x0, seeds[i]->y0, seeds[i]->w,
                    seeds[i]->h)).clone();
        Size whole;
        Point ref;
        img0_roi.locateROI(whole, ref);
        cout << whole << " "<< ref << endl;
        cout << "sub " << img0_roi.isSubmatrix() << endl;

        namedWindow("roi");
        imshow("roi", img0_roi);
        waitKey(0);
        GaussianBlur(img0_roi, g1, Size(31,31), 0);
        //GaussianBlur(img0, g1, Size(31,31), 0);
        GaussianBlur(img0_roi, g2, Size(3,3), 0);
        //GaussianBlur(img0, g2, Size(3,3), 0);
        Mat img0_roi_dog = g2 - g1; //NOTE: CPP overflow handle
        imwrite("roi_dog.tif", img0_roi_dog);
        namedWindow("roi_dog");
        imshow("roi_dog", img0_roi_dog);
        waitKey(0);
        // Find bigest connective graph
        //connectedCompoents(img0_roi_dog, img0_roi_dog_cc, );
        // Recenter wormbody in image
        //
        /*
        wormbody worm_x;
        worm_x.x = ;
        worm_x.y =;
        worm_x.body =;
        worm_x.worm_label=;
        worm_x.worm_frame=;

        vector<wormbody *> worm_t;
        worm_t.push_back( &worm_x );
        worms.push_back( &worm_t );
        */
    }
    
}


int main(int argc, char *argv[]){
    string imgs_dirs = "";
    string out_dirs = "";
    string is_auto = "";
   
    // Accept arguments
    if (argc < 3){
        help();
    }else{
        imgs_dirs = argv[1];
        out_dirs = argv[2];
        if (argc > 3){
            is_auto = argv[3];
        }
        // Only accept front 3 arguments
    }
    
    // Fetch all images
    // error event handle: file type detetion
    glob(imgs_dirs, filenames);
    int imgs_num = filenames.size();
    int worm_num = 0; // inite to zero
    
    // Initial start position by auto or manual
    Mat img0 = imread(filenames[0], -1);
    cout << is_auto << endl;
    tracker_init(img0, is_auto);
        
    #pragma omp parallel for
    for (int i = 0; i < imgs_num; ++i)
    {
        cout << i <<endl;
        Mat whole_img = imread( filenames[i], -1);
        //#pragma omp parallel for
        // For each worm do track
        for (int worm = 0; worm < worm_num; ++worm)
        {
            
        }

    }
    
    return 0;
}

void help(){
    cout << "Usage: tracking <images_source_dir> <result_dir> <operation>" << endl;
    cout << "operations:" << endl;
    cout << "\ttracking {-h --help}" << endl;
    cout << "\ttracking {-a --auto}" << endl;
    cout << "\tMore than 3 arguments would be ignored" <<endl;
}

// Recenter wormbody in image, then return image
void tracker_update(Mat whole_img, Mat body_new, unsigned int x_ori_old,
        unsigned int y_ori_old, unsigned int x_ori_new, unsigned y_ori_new)
{

    int rows = body_new.rows;
    int cols = body_new.cols;
    int top, bottom = -1;
    int left = cols;
    int right = -1;
    // Find edge at 4 diretions
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j< cols; ++j){
            if ( body_new(i, j) > 0 ){
                if( top == -1){
                    top = i;
                }else{
                    bottom = i;
                }
                if( j < left ){
                    left = j;
                }else if( j > right ){
                    right = j;
                }
            }
        }
    }
    int x_mid, y_mid = 0;
    if( left < right && top < bottom ){
        x_mid = (left + right)/2;
        y_mid = (top + bottom)/2;
    }
    Mat worm_body_mask = Mat.zeros(Size(window, window), CV_16U);
    body_new 
}
