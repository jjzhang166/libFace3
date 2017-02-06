#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

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

class Classifier {
public:
    Classifier(const std::string& model_file, const std::string& trained_file, const std::string& mean_file, int cpu_only);
    Classifier(const std::string& model_file, const std::string& trained_file, int cpu_only);
    ~Classifier();
    std::vector<float> get_feature(const cv::Mat& img);
    std::vector<float> get_feature(const std::string& image_name);
    void SetMean(const std::string& mean_file);
    dlib::frontal_face_detector detector;
    dlib::shape_predictor sp;
private:

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

private:
    std::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    bool flag_sub_mean_ = false;
    int scale_ = 1;
};

Classifier::~Classifier() {

}

Classifier::Classifier(const std::string& model_file,
                       const std::string& trained_file,
                       const std::string& mean_file, int cpu_only) {
    flag_sub_mean_ = true;
    if (cpu_only == 1) {
        Caffe::set_mode(Caffe::CPU);
    }
    else {
        Caffe::set_mode(Caffe::GPU);
    }

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";

    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    SetMean(mean_file);
    Blob<float>* output_layer = net_->output_blobs()[0];
    detector = dlib::get_frontal_face_detector();
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
}

Classifier::Classifier(const std::string& model_file, const std::string& trained_file, int cpu_only) {
    flag_sub_mean_ = false;
    if (cpu_only == 1) {
        Caffe::set_mode(Caffe::CPU);
    }
    else {
        Caffe::set_mode(Caffe::GPU);
    }
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    //get the address of Point
    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";

    //get size
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    Blob<float>* output_layer = net_->output_blobs()[0];
    detector = dlib::get_frontal_face_detector();
    dlib::deserialize("./resource/shape_predictor_68_face_landmarks.dat") >> sp;

}

void Classifier::SetMean(const string& mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_) << "Number of channels of mean file doesn't match input layer.";
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }
    cv::Mat mean;
    cv::merge(channels, mean);
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::get_feature(const cv::Mat& img) {
    double time1;
    Blob<float>* input_layer = net_->input_blobs()[0];

    input_layer->Reshape(1, num_channels_,
        input_geometry_.height, input_geometry_.width);
    net_->Reshape();

    std::vector<cv::Mat> input_channels;

    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);
    net_->Forward();

    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();

    return std::vector<float>(begin, end);
}

std::vector<float> Classifier::get_feature(const string& image_name) {
    double time1;
    cv::Mat img = cv::imread(image_name, -1);
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels);
    net_->Forward();

    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}

void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}


void Classifier::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    if (flag_sub_mean_ == true) {
        cv::subtract(sample_float, mean_, sample_normalized);
        cv::split(sample_normalized, *input_channels);
    }
    else {
        cv::split(sample_float, *input_channels);
    }

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

#endif // CLASSIFIER_HPP
