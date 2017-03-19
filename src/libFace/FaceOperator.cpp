#include "libFace/FaceOperator.h"
#include "libFace/Log.h"
#include "libFace/Classifier.h"
#include "libFace/Utility.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <mutex>
#include <vector>


namespace libface {


/* below is static variable define */
std::mutex FaceOperator::_classierMutex;
cv::CascadeClassifier FaceOperator::_cascadeClassier;
Classifier FaceOperator::_classifier = Classifier(std::string("./resource/deploy.prototxt"),
                                                  std::string("./resource/face.model"), 1);
std::mutex FaceOperator::_classifierMutex;
float FaceOperator::_maxDist = 800;
float FaceOperator::_distScale = 0.00087;
float FaceOperator::_distScale2 = 0.6;

/* below is static function define */

bool FaceOperator::LoadCascadeClassifier(const std::string &fliePath) {
    return _cascadeClassier.load(fliePath);
}

void FaceOperator::FaceDetect(cv::Mat img, double scale, bool tryflip, DetectResHandle resultHandle) {
    using namespace cv;
    using namespace std;
    try {
        if(img.empty()) {
            LOG_INFO("img is empty\n");
            return;
        }

        int i = 0;
        vector<Rect> faces, faces2;
        faces.clear();
        faces2.clear();

        const static Scalar colors[] = { CV_RGB(0, 0, 255),
                                         CV_RGB(0, 128, 255),
                                         CV_RGB(0, 255, 255),
                                         CV_RGB(0, 255, 0),
                                         CV_RGB(255, 128, 0),
                                         CV_RGB(255, 255, 0),
                                         CV_RGB(255, 0, 0),
                                         CV_RGB(255, 0, 255) };

        Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
        cvtColor(img, gray, CV_BGR2GRAY);
        resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
        equalizeHist(smallImg, smallImg);

        if(!_classierMutex.try_lock()) {
            cout << "CascadeClassie is working" << endl;
            return;
        }
        _cascadeClassier.detectMultiScale(smallImg, faces,
                                           1.1, 2, 0
                                           //|CV_HAAR_FIND_BIGGEST_OBJECT
                                           //|CV_HAAR_DO_ROUGH_SEARCH
                                           | CV_HAAR_SCALE_IMAGE,
                                           Size(30, 30));
        if (tryflip) {
            flip(smallImg, smallImg, 1);
            _cascadeClassier.detectMultiScale(smallImg, faces2,
                                               1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
                                               Size(30, 30));
            cout << "faces2.size():" << faces2.size() << endl;
            for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++) {
                faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
            }
        }
        _classierMutex.unlock();

        if (faces.size() > 0) {
            LOG_INFO("~~~~~~~~~~~~~~faces count: %d~~~~~~~~~~~~~~\n", faces.size());
            for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++) {
                Mat procImage(smallImg, Rect((r->x)*scale, r->y*scale, r->width*scale, r->height*scale));
                // handle detect result
                if(resultHandle) {
                    resultHandle(procImage);
                } else {
                    LOG_INFO("face detect not result handler\n");
                    continue;
                }
            }

        } else {
            LOG_INFO("This image is not a face, ignore it\n");
        }

    }
    catch (std::exception& e) {
        LOG_INFO("exception thrown->[%s]\n", e.what());
    }
}

float FaceOperator::CalculateDistance(const cv::Mat& feature1, const cv::Mat& feature2, const int distFlag) {
    float dist = 0.0;
    if (distFlag == 1) {
        dist = cv::norm(feature1, feature2, cv::NORM_L2);
        //dist = 1 / (1 + dist);
    } else if (distFlag == 2) {
        float ab = feature1.dot(feature2);
        float aa = feature1.dot(feature1);
        float bb = feature2.dot(feature2);
        dist = ab / sqrt(aa*bb+0.00001);
        //dist = 1 - dist;                //test, not sure;
        dist = (1 + dist) / 2;
    }
    return dist;
}

float FaceOperator::ProcessSimilarity(float dist) {
    float oldDist = dist;
    float T = 0.75;
    float similarity = dist + (dist - T) * _distScale2;
    if (similarity>1) {
        similarity = 1;
    };
    if (similarity < 0.01) {
        similarity = 0.01;
    }
    if (oldDist != 1 && oldDist>T) {
        similarity = similarity - Randf() / 100;
    }
    return similarity;
}

void FaceOperator::LinearSearch(const cv::Mat& data, const cv::Mat& point,
                   cv::Mat& indices, cv::Mat& dists, const int k, int distFlag) {
    if (distFlag == 1) {
        cv::flann::Index flannIndex(data, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_EUCLIDEAN);
        flannIndex.knnSearch(point, indices, dists, k, cv::flann::SearchParams(64));
    } else {
        cv::flann::Index flannIndex(data, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_L2);
        flannIndex.knnSearch(point, indices, dists, k, cv::flann::SearchParams(64));
    }
}

void FaceOperator::KnnSearch(const cv::Mat& data, const cv::Mat& point,
                std::vector<int>& indices, std::vector<float>& dists, const int k, int distFlag) {
    cv::vector<float> tempDist(data.rows,0);
    #pragma omp parallel for
    for (int i = 0; i < data.rows; i++) {
        //tempDist.push_back(calculate_distance(data.row(i), point, distFlag));
        tempDist[i] = CalculateDistance(data.row(i), point, distFlag);
    }
    if (distFlag == 1) {	// 欧式距离，从小到大排列
        std::size_t n(0);
        std::generate(std::begin(indices), std::end(indices), [&]{ return n++; });
        std::sort(std::begin(indices), std::end(indices), [&](int i1, int i2) { return tempDist[i1] < tempDist[i2]; });
       #pragma omp parallel for
        for (int i = 0; i < data.rows; i++) {
            dists[i] = tempDist[indices[i]];
        }
    }
    if (distFlag == 2) {
        std::size_t n(0);
        std::generate(std::begin(indices), std::end(indices), [&]{ return n++; });
        std::sort(std::begin(indices), std::end(indices), [&](int i1, int i2) { return tempDist[i1] > tempDist[i2]; });
       #pragma omp parallel for
        for (int i = 0; i < data.rows; i++) {
            dists[i] = tempDist[indices[i]];
        }
    }

}

int FaceOperator::DlibImg2Mat(dlib::array2d<dlib::rgb_pixel>&in, cv::Mat &out) {
    cv::Mat tempM;
    tempM = dlib::toMat(in);
    cv::Mat dstTypeM = cv::Mat(tempM.rows, tempM.cols, CV_32F);
    tempM.convertTo(tempM, CV_32F);
    out = tempM.clone();
    if (out.dims > 1) {		// if dims greats than 1, we say that out has valid data.
        return 1;
    } else {
        return 0;
    }
}

int FaceOperator::GetNormalizationFace(const char* inputFileName, cv::Mat& face, int targetWidth) {
    using namespace std;

    if (targetWidth <= 20) {
        targetWidth = 20;
    } else if (targetWidth > 256) {
        targetWidth = 256;
    }
    try {
        dlib::array2d<dlib::rgb_pixel> img;
        load_image(img, inputFileName);
        int scaleCount = 0;
        dlib::pyramid_down<2> pyr;
        dlib::array2d<dlib::rgb_pixel> temp;
        int picWidth = img.nc();
        int picHeight = img.nr();
        int origWidth = picWidth;
        int origHeight = picHeight;
        double scaleFactor = 1.0;

//        while ((picWidth*picHeight) <= (80 * 80) && scaleCount++ < 1) {
//            pyramid_up(img);
//            picWidth = img.nc();
//            picHeight = img.nr();
//        }
//        while ((picWidth*picHeight) > (1000 * 640 * 1)) {
//            pyr(img, temp);
//            swap(temp, img);
//            picWidth = img.nc();
//            picHeight = img.nr();
//        }

        scaleFactor = picWidth / origWidth;

        _classifierMutex.lock();
        std::vector<dlib::rectangle> dets;
        dets = _classifier.detector(img);
        _classifierMutex.unlock();

        sched_yield();

        long maxWidth = 0;
        long index = -1;
        for (unsigned long j = 0; j < dets.size(); ++j) {		//find the biggest face
            if (dets[j].width() > maxWidth) {
                maxWidth = dets[j].width();
                index = j;
            }
        }
        if (index < 0) {
            return -1;
        }
        if (scaleFactor > 1.0) {		// resume the size to original face
            load_image(img, inputFileName);
            dets[index].set_top((long)((double)(dets[index].top()) / scaleFactor));
            dets[index].set_bottom((long)((double)(dets[index].bottom()) / scaleFactor));
            dets[index].set_right((long)((double)(dets[index].right()) / scaleFactor));
            dets[index].set_left((long)((double)(dets[index].left()) / scaleFactor));
        }
        std::vector<dlib::full_object_detection> shapes;
        if (index >= 0) {
            dlib::full_object_detection shape;
            _classifierMutex.lock();
            shape = _classifier.sp(img, dets[index]);
            _classifierMutex.unlock();
            shapes.push_back(shape);
        }
        if(0 == shapes.size()) {
            cout << "image's point is null!" << endl;
            return -2;
        }
        dlib::array<dlib::array2d<dlib::rgb_pixel> > face_chips;
        dlib::extract_image_chips(img, get_face_chip_details(shapes, targetWidth, 0.2), face_chips);
        cv::Mat tempFace;
        DlibImg2Mat(face_chips[0], tempFace);
        face = tempFace.clone();
    } catch (std::exception& e) {
        std::cout << "\nexception thrown!" << std::endl;
        std::cout << e.what() << std::endl;
        return -4;
    }

    return 0;
}

int FaceOperator::ExtractFeature(const std::string& imagePath, std::vector<float>& feature) {
    using namespace std;
    try {
        int imgSize = 128;
        int feature_dim = 256;
        int scale = 1;

        string extName = GetExtName(imagePath);
        if (!IsImg(extName)) {
            return -101; // invalid picture
        }
        cv::Mat face;
        int ret = 0;
        if ((ret = GetNormalizationFace(imagePath.c_str(), face, imgSize)) != 0) {
            std::cout << "No face is detected in the image" << imagePath << std::endl;
            return ret;
        }
        if (scale == 1) {
            face = face / 255;
        }
        _classifierMutex.lock();
        _classifier.GetFeature(face).swap(feature);
        _classifierMutex.unlock();
    } catch (std::exception& e) {
        std::cout << "\nexception thrown!" << std::endl;
        std::cout << e.what() << std::endl;
        return -102;
    }
}

void FaceOperator::FaceRecognition(cv::Mat recognizeFeature, cv::Mat srcFeature, float &similary) {
    std::vector<int> indices(srcFeature.rows);
    std::vector<float> dists(srcFeature.rows);
    int k;
    KnnSearch(srcFeature, recognizeFeature, indices, dists, k, 2);
    similary = dists[0];
}

void FaceOperator::FaceRecognition(cv::Mat recognizeFeature, const std::vector<cv::Mat>& featureSet,
                                   std::vector<int>& indices, std::vector<float>& dists) {
    cv::Mat featureSetTmp = cv::Mat::zeros(featureSet.size(), 256, CV_32FC1);
    for(int i = 0; i < featureSet.size(); ++i) {
        for(int j = 0; j < featureSet[i].cols; ++j) {
            featureSetTmp.at<float>(i, j) = featureSet[i].at<float>(0, j);
        }
    }
    int k;
    KnnSearch(featureSetTmp, recognizeFeature, indices, dists, k, 2);
}



}
