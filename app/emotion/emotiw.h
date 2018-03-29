#ifndef EMOTIW_H
#define EMOTIW_H

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/shared_ptr.hpp>

enum class Emotion {Angry , Disgust , Fear , Happy, Neutral, Sad, Surprise};
using EmAndConf = std::vector<std::pair<Emotion, float> >;
class EmotiW
{
public:
    EmotiW(const std::string prototxt, const std::string caffemodel, const int gpu_id);
    EmAndConf GetEmotion(const cv::Mat image);
    void SetDevice(const int gpu_id);
private:
    std::shared_ptr<caffe::Net<float> > net_;
    boost::shared_ptr<caffe::MemoryDataLayer<float> > input_;
    std::vector<cv::Mat> images_;
    EmAndConf e_;
    std::vector<int> lables_ = {1};
};

#endif // EMOTIW_H
