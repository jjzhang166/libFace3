#include "FaceOperator.hpp"
#include "CameraOperator.hpp"

#include <opencv2/core.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thread>

const int STRING_FILENAME_LENGTH = 128;
static int indexFile = 0;

void detectResultHandle(cv::Mat res){
    //Mat saveImage(smallImg, Rect((smallImg.cols - r->x - r->width)*scale, r->y*scale, r->width*scale, r->height*scale));
    char fileName[STRING_FILENAME_LENGTH]{0};
    snprintf(fileName, STRING_FILENAME_LENGTH, "./image/test%d.jpg", indexFile);
    indexFile++;
    cv::imwrite(fileName, res);

    std::thread show([&](){
        cv::imshow("abc", res);
        cv::waitKey(1);
    });
    show.join();
}


int main(int argc, char *argv[])
{
    bool loadResult = FaceOperator::loadCascadeClassifier("./resource/haarcascade_frontalface_alt.xml");
    if(false == loadResult){
        std::cerr << "load cascadfliePathe classier failed\n";
        exit(EXIT_FAILURE);
    }

    CameraOperator::Handler handle = [](cv::Mat mat){
        cv::imshow("use camera", mat);
        cv::waitKey(1);
        std::thread task(&FaceOperator::faceDetect, mat, 1.0, true, detectResultHandle);
        task.detach();
    };

    CameraOperator::handleFrameFromUSBCamera(handle);

    return 0;
}
