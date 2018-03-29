#include "emotiw.h"
#include <algorithm>

using namespace caffe;
using namespace std;

EmotiW::EmotiW(const string prototxt, const string caffemodel, const int gpu_id)
{
    FLAGS_minloglevel = google::FATAL;
    SetDevice(gpu_id);
    net_.reset(new Net<float>(prototxt, Phase::TEST));
    net_->CopyTrainedLayersFrom(caffemodel);
    e_.reserve(7);
}

EmAndConf EmotiW::GetEmotion(const cv::Mat image)
{
    e_.clear();
    images_.push_back(image);
    float loss = 0.0;
    input_ = static_pointer_cast<MemoryDataLayer<float> >(net_->layer_by_name("data"));
    input_->AddMatVector(images_, lables_);

    const vector<Blob<float>* >& results = net_->Forward(&loss);
    const float* data = results[1]->mutable_cpu_data();
    for (int i = 0; i < 7 ; i++)
    {
        Emotion e = static_cast<Emotion>(i);
        e_.push_back(make_pair(e,data[i]));
    }
    //const vector<float> emotions {data, data + 7};
    //auto max_id = max_element(emotions.cbegin(), emotions.cend());
    //e_ = static_cast<Emotion>(distance(emotions.cbegin(), max_id));
    return e_;
}

void EmotiW::SetDevice(const int gpu_id)
{
    if (gpu_id < 0)
    {
        Caffe::set_mode(Caffe::CPU);
    }
    else
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(gpu_id);
    }

}

