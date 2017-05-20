#ifndef _LIBFACE_FaceOperator_H
#define _LIBFACE_FaceOperator_H

#include "libFace/Classifier.h"

#include <opencv2/core.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/to_open_cv.h>

#include <functional>
#include <mutex>


namespace libface {


class FaceOperator {
public:
    typedef std::function<void(cv::Mat)> DetectResHandle;

public:
    static bool LoadCascadeClassifier(const std::string& fliePath);
    static void FaceDetect(cv::Mat img, double scale, bool tryflip, DetectResHandle resultHandle);
    static int ExtractFeature(const std::string& imagePath, std::vector<float>& feature);
    static void FaceRecognition(cv::Mat recognizeFeature, cv::Mat srcFeature, float& similary);
    static void FaceRecognition(cv::Mat recognizeFeature, const std::vector<cv::Mat>& featureSet,
                                   std::vector<int>& indices, std::vector<float>& dists);

private:
    static inline double Randf() {
        return (double)(rand() / (double)RAND_MAX); // produce 0 ~ 1 float number.
    }
    static float CalculateDistance(const cv::Mat& feature1, const cv::Mat& feature2, const int distFlag);
    static float ProcessSimilarity(float dist);
    static void LinearSearch(const cv::Mat& data, const cv::Mat& point,
                       cv::Mat& indices, cv::Mat& dists, const int k, int distFlag);
    static void KnnSearch(const cv::Mat& data, const cv::Mat& point,
                    std::vector<int>& indices, std::vector<float>& dists, const int k, int distFlag);

    static int DlibImg2Mat(dlib::array2d<dlib::rgb_pixel>& in, cv::Mat &out);
    static int GetNormalizationFace(const char* inputFileName, cv::Mat& face, int targetWidth);

private:
    static std::mutex				_classierMutex;
    static cv::CascadeClassifier	_cascadeClassier;

    static Classifier				_classifier;
    static std::mutex				_classifierMutex;

    // TODO: 临时放置
    static float					_maxDist;
    static float					_distScale;
    static float					_distScale2;

};




}	// namespace libface


#endif // _LIBFACE_FaceOperator_H
