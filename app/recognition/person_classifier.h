#ifndef PERSONCLASSIFIER_H
#define PERSONCLASSIFIER_H

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>

using namespace caffe;
class PersonClassifier
{
public:
    PersonClassifier(const std::string prototxt_path, const std::string model_path, const int gpuid);
    std::vector<float> Detect(const cv::Mat& face);
    void setOutput(const int output);
    void setInput(const cv::Size target);

private:
    std::shared_ptr<Net<float> > net_;
    int number_of_features_;
    cv::Size target_;
};

#endif // PERSONCLASSIFIER_H
