#include <omp.h>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
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


int main(int argc, char *argv[]){
    string imgs_dirs = "";
    string out_dirs = "";
    bool is_auto = false;
    
    if (argc < 3){
        help();
    }else{
        imgs_dirs = argv[1];
        out_dirs = argv[2];
        if (argc == 4){
            is_auto = true;
        } // If more than 4 argument, just ignore it
    }
    
    // Fetch all images
    // error event handle: file type detetion
    glob(imgs_dirs, filenames);    
    
    // initial start position by auto or manual

    #pragma omp parallel for
    for (int i = 0; i < filenames.size(); ++i)
    {
        cout << i <<endl;
        Mat whole_img = imread( filenames[i], -1);
        //#pragma omp parallel for

    }
    
    cout << endl;
    return 0;
}

void help(){
    cout << "Usage: tracking <images_source_dir> <result_dir> <operation>" << endl;
    cout << "operations:" << endl;
    cout << "\ttracking {-h --help}" << endl;
    cout << "\ttracking {-a --auto}" << endl;
    cout << "\tMore than 3 arguments would be ignored" <<endl;
}
