#include "libFace/Classifier.h"

using namespace caffe;


namespace libface {


Classifier::~Classifier() {

}

Classifier::Classifier(const std::string& model_file,
                       const std::string& trained_file,
                       const std::string& mean_file, int cpu_only) {
    _flag_sub_mean = true;
    if (cpu_only == 1) {
        Caffe::set_mode(Caffe::CPU);
    }
    else {
        Caffe::set_mode(Caffe::GPU);
    }

    /* Load the network. */
    _net.reset(new Net<float>(model_file, TEST));
    _net->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(_net->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(_net->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = _net->input_blobs()[0];
    _num_channels = input_layer->channels();
    CHECK(_num_channels == 3 || _num_channels == 1)
        << "Input layer should have 1 or 3 channels.";

    _input_geometry = cv::Size(input_layer->width(), input_layer->height());
    SetMean(mean_file);
    Blob<float>* output_layer = _net->output_blobs()[0];
    detector = dlib::get_frontal_face_detector();
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
}

Classifier::Classifier(const std::string& model_file, const std::string& trained_file, int cpu_only) {
    _flag_sub_mean = false;
    if (cpu_only == 1) {
        Caffe::set_mode(Caffe::CPU);
    }
    else {
        Caffe::set_mode(Caffe::GPU);
    }
    _net.reset(new Net<float>(model_file, TEST));
    _net->CopyTrainedLayersFrom(trained_file);
    CHECK_EQ(_net->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(_net->num_outputs(), 1) << "Network should have exactly one output.";

    //get the address of Point
    Blob<float>* input_layer = _net->input_blobs()[0];
    _num_channels = input_layer->channels();
    CHECK(_num_channels == 3 || _num_channels == 1) << "Input layer should have 1 or 3 channels.";

    //get size
    _input_geometry = cv::Size(input_layer->width(), input_layer->height());
    Blob<float>* output_layer = _net->output_blobs()[0];
    detector = dlib::get_frontal_face_detector();
    dlib::deserialize("./resource/shape_predictor_68_face_landmarks.dat") >> sp;

}

void Classifier::SetMean(const string& mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), _num_channels) << "Number of channels of mean file doesn't match input layer.";
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < _num_channels; ++i) {
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }
    cv::Mat mean;
    cv::merge(channels, mean);
    cv::Scalar channel_mean = cv::mean(mean);
    _mean = cv::Mat(_input_geometry, mean.type(), channel_mean);
}

std::vector<float> Classifier::GetFeature(const cv::Mat& img) {
    double time1;
    Blob<float>* input_layer = _net->input_blobs()[0];

    input_layer->Reshape(1, _num_channels,
        _input_geometry.height, _input_geometry.width);
    _net->Reshape();

    std::vector<cv::Mat> input_channels;

    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);
    _net->Forward();

    Blob<float>* output_layer = _net->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();

    return std::vector<float>(begin, end);
}

std::vector<float> Classifier::GetFeature(const std::string& image_name) {
    double time1;
    cv::Mat img = cv::imread(image_name, -1);
    Blob<float>* input_layer = _net->input_blobs()[0];
    input_layer->Reshape(1, _num_channels, _input_geometry.height, _input_geometry.width);
    _net->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels);
    _net->Forward();

    Blob<float>* output_layer = _net->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}

void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = _net->input_blobs()[0];
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
    if (img.channels() == 3 && _num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && _num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && _num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && _num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != _input_geometry)
        cv::resize(sample, sample_resized, _input_geometry);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (_num_channels == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    if (_flag_sub_mean == true) {
        cv::subtract(sample_float, _mean, sample_normalized);
        cv::split(sample_normalized, *input_channels);
    }
    else {
        cv::split(sample_float, *input_channels);
    }

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == _net->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}


}	// namespace libface
