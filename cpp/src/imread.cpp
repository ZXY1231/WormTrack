#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

/******************************************************************************
 * Optimize Image Acess Speed
 * | Version | Commit
 * |   0.1   | H.F 20190706
 *
 * Idea: Is it possible to access camera to anlysis data in real time 
 ******************************************************************************/

int main(int argc, char** argv )
{

    vector<String> filenames; // notice here that we are using the Opencv's embedded "String" class
    //String folder = "/igem/Results/20190404/Worm_190404_121131/0-04091621/body"; // again we are using the Opencv's embedded "String" class
    String folder = argv[1];
    glob(folder, filenames); // new function that does the job ;-)

    for(size_t i = 0; i < filenames.size(); ++i)
    {
        //Mat src = imread(filenames[i]);
        cout << filenames[i] << endl;
        //if(!src.data)
        //   cerr << "Problem loading image!!!" << endl;

        /* do whatever you want with your images here */
    }
}
