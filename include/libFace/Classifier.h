#ifndef _LIBFACE_CLASSIFIER_HPP
#define _LIBFACE_CLASSIFIER_HPP

#include <caffe/caffe.hpp>

#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/to_open_cv.h>

#include <memory>
#include <string>

using namespace caffe;

namespace libface {

class Classifier {
public:
    Classifier(const std::string& model_file, const std::string& trained_file,
               const std::string& mean_file, int cpu_only);
    Classifier(const std::string& model_file, const std::string& trained_file,
               int cpu_only);
    ~Classifier();
    std::vector<float> GetFeature(const cv::Mat& img);
    std::vector<float> GetFeature(const std::string& image_name);
    void SetMean(const std::string& mean_file);
    dlib::frontal_face_detector detector;
    dlib::shape_predictor sp;

private:
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

private:
    std::shared_ptr<Net<float>>				_net;
    cv::Size								_input_geometry;
    int										_num_channels;
    cv::Mat									_mean;
    bool									_flag_sub_mean;
    int										_scale;
};



}	// namespace libface


#endif // CLASSIFIER_HPP
