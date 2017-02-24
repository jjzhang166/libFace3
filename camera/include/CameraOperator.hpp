#ifndef CameraOperator_H
#define CameraOperator_H

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <functional>

#include <sys/types.h>
#include <sys/unistd.h>


class CameraOperator {
public:
    /* type define */
    typedef std::function<void(cv::Mat)> Handler;

    /* prototype */
    static void HandleFrameFromRtspCamera(std::string rtspAddress, Handler handler);
    static void HandleFrameFromUSBCamera(Handler handle);
    static void HandleFrameFromImage(std::string imagePath, Handler handle);
};

/* below is static function define */

void CameraOperator::HandleFrameFromRtspCamera(std::string rtspAddress, Handler handler) {
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

void CameraOperator::HandleFrameFromUSBCamera(Handler handler) {
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

void CameraOperator::HandleFrameFromImage(std::string imagePath, Handler handler) {
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


#endif // CameraOperator_H
