#include "camera/CameraOperator.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <functional>

#include <sys/types.h>
#include <sys/unistd.h>


namespace libface {

namespace camera {

void HandleFrameFromRtspCamera(const std::string& rtspAddress, Handler handler) {
//    string rtspAddress = "rtsp://admin:qy38888813@192.168.1.64:554/che/main/av_stream";
    using namespace cv;
    using namespace std;

    Mat image;
    VideoCapture vcap;
    vcap.open(rtspAddress);
    //open the usb camera and make sure it's opened
    if (!vcap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return;
    }

    while (true) {
        if (!vcap.read(image)) {
            cout << "No frame" << endl;
            continue;
        }
        waitKey(1);
        Mat faceImage = image.clone();
        if(handler != NULL) {
            handler(faceImage);
        }
    }

}

void HandleFrameFromUSBCamera(Handler handler) {
    using namespace cv;
    using namespace std;

    Mat image;
    VideoCapture vcap;
    vcap.open(0);
    //open the usb camera and make sure it's opened
    if (!vcap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return;
    }

    while (true) {
        if (!vcap.read(image)) {
            cout << "No frame" << endl;
            continue;
        }
        Mat faceImage = image.clone();
        if(handler != NULL) {
            handler(faceImage);
        }
    }

}

void HandleFrameFromImage(const std::string& imagePath, Handler handler) {
    using namespace cv;
    using namespace std;
    Mat image = imread(imagePath);
    if(!image.data) {
        cout << "No data!--Exiting the program" << endl;
        return;
    }
    if(handler != NULL) {
        handler(image);
    }
//    waitKey(5000);

}

}	// namespace camera

}	// namespace libface



