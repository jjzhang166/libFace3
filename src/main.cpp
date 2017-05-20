#include "libFace/FaceOperator.h"
#include "libFace/Log.h"
#include "camera/CameraOperator.h"

//#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thread>

const int STRING_FILENAME_LENGTH = 128;
static int indexFile = 0;

void DetectResultHandle(cv::Mat res) {
    using libface::FaceOperator;

    char fileName[STRING_FILENAME_LENGTH]{0};
    snprintf(fileName, STRING_FILENAME_LENGTH, "./image/test%d.jpg", indexFile);
    indexFile++;
    cv::imwrite(fileName, res);

    timeval start, end;
    gettimeofday(&start, NULL);

    std::vector<float> feature;
    cv::Mat srcFeature = cv::Mat::zeros(1, 256, CV_32FC1);
    int ret = FaceOperator::ExtractFeature(fileName, feature);
    unlink(fileName);

    LOG_INFO("return code: %d\n", ret);
    if(ret == 0){
        LOG_INFO("Get the face feature, feature cols: %d.\nfeature code: \n", feature.size());
        for(int i = 0; i < feature.size(); ++i){
            srcFeature.at<float>(0, i) = feature[i];
            if(i %  10) {
                std::cout << feature[i] << " ";
            } else {
                std::cout << "\n";
            }
        }
        std::cout << "\n";
    }

    gettimeofday(&end, NULL);
    int tmp = 1000000;
    long double payTime = (tmp * (end.tv_sec - start.tv_sec)) + (end.tv_usec - start.tv_usec);
    std::cerr << "extract feature pay time is:" <<  payTime / tmp << " s\n";

    float simi = -1;
    FaceOperator::FaceRecognition(srcFeature, srcFeature, simi);
    std::cerr << "face recognition similary: " << simi << "\n";

    std::thread show([&]() {
        cv::imshow("abc", res);
        cv::waitKey(1);
    });
    show.join();
}

int main(int argc, char *argv[]) {
    using namespace libface::camera;
    using libface::FaceOperator;

    bool loadResult = FaceOperator::LoadCascadeClassifier("./resource/haarcascade_frontalface_alt.xml");
    if(false == loadResult) {
        std::cerr << "load cascadfliePathe classier failed\n";
        exit(EXIT_FAILURE);
    }

    Handler handle = [](cv::Mat mat) {
        cv::imshow("use camera", mat);
        cv::waitKey(1);
        std::thread task(&FaceOperator::FaceDetect, mat, 1.0, true, DetectResultHandle);
        task.detach();
    };

    HandleFrameFromUSBCamera(handle);
    return 0;
}
