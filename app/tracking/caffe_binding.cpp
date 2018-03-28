#include "caffe_binding.h"

using namespace caffe;
using namespace std;

CaffeBinding::CaffeBinding()
{
    FLAGS_minloglevel = google::FATAL;
}

int CaffeBinding::AddNet(string model_definition, string weights, int gpu_id) {

    SetDevice(gpu_id);
    auto new_net = make_shared<Net<float> >(model_definition, Phase::TEST);
    new_net->CopyTrainedLayersFrom(weights);
    nets_.push_back(new_net);
    prototxts_.push_back(model_definition);
    return nets_.size() - 1;
}


Data caffe::CaffeBinding::Forward(int net_id)
{
    auto predictor = nets_[net_id];
    const vector<Blob<float>* >& nets_output = predictor->ForwardPrefilled();

    Data result;
    for (int n = 0; n < nets_output.size(); n++)
    {
        DataBlob blob = { nets_output[n]->cpu_data(), nets_output[n]->shape(),
                          predictor->blob_names()[predictor->output_blob_indices()[n]] };
        result[blob.name] = blob;
    }
    return result;
}

Data CaffeBinding::Forward(VecMat&& input_image, int net_id)
{
    SetMemoryDataLayer("data", move(input_image), net_id);
    return Forward(net_id);
}

void caffe::CaffeBinding::SetMemoryDataLayer(string layer_name, VecMat&& input_image, int net_id)
{
    auto predictor = nets_[net_id];
    vector<int> labels;
    labels.push_back(1);
    auto data_layer_ptr = static_pointer_cast<MemoryDataLayer<float>, Layer<float>>(predictor->layer_by_name(layer_name));
    data_layer_ptr->AddMatVector(input_image, labels);
}

void CaffeBinding::SetDevice(int gpu_id) {
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

CaffeBinding::~CaffeBinding()
{
}
