#include <omp.h>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include "main.h"

using namespace std;
using namespace cv;

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
    unsigned int worm_key;
    unsigned int worm_frame;
};


vector<wormbody *> worm_t;
vector<vector<wormbody> *> worms;

void tracker_init(Mat img0, vector<vector<wormbody>*> *worms_l, string AUTO){
    
    if (AUTO.length() > 0){ // AUTO model: load initial position from outside list

    }else{ // select one worm by hand
        Rect2d r = selectROI(img0);
        cout << r.x <<" " << r.y <<" "<< r.width <<" "<< r.height  << endl;
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
    tracker_init(img0, &worms, is_auto);
        
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
