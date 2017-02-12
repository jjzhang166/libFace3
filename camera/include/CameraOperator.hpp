#ifndef CameraOperator_H
#define CameraOperator_H

#include "HCNetWrapper.hpp"

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
    static void handleFrameFromRtspCamera(std::string rtspAddress, Handler handler);
    static void handleFrameFromUSBCamera(Handler handle);
    static void handleFrameFromHCNetSDK(HCNetWrapper& hcNet, Handler handle);
    static void handleFrameFromImage(std::string imagePath, Handler handle);
};

/* below is static function define */

void CameraOperator::handleFrameFromRtspCamera(std::string rtspAddress, Handler handler) {
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


void CameraOperator::handleFrameFromUSBCamera(Handler handler) {
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

void CameraOperator::handleFrameFromHCNetSDK(HCNetWrapper& hcNet, Handler handler) {
    using namespace cv;
    using namespace std;
    try {
        LONG lUserID = hcNet.loginV30();
        if (lUserID < 0) {
            cout << "Login Error---" << hcNet.getLastError() << endl;
            return;
        }

        // start preview and callBack stream
        LONG lRealPlayHandle;
        hcNet.setPlayInfo(0, 1, 0, 0);
        lRealPlayHandle = hcNet.realPlayV40();

        cout << "lRealPlayHandle:" << lRealPlayHandle << endl;
        if(lRealPlayHandle < 0) {
            cout << "NET_DVR_RealPlay_V40 error---" << hcNet.getLastError() << endl;
            hcNet.loginOut();
        }
        waitKey();
        sleep(-1); //wangben - TODO

        hcNet.stopRealPlay(lRealPlayHandle);
        hcNet.loginOut();
        return;
    } catch (exception& e) {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
        throw;
    }
}

void CameraOperator::handleFrameFromImage(std::string imagePath, Handler handler) {
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
