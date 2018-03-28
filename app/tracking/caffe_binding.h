#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

namespace caffe {
    struct DataBlob {
        const float* data;
        std::vector<int> size;
        std::string name;
    };

    using Data = std::unordered_map<std::string, DataBlob>;
    using VecMat = std::vector<cv::Mat>;
    class CaffeBinding
    {
    public:
        CaffeBinding();
        int AddNet(std::string prototxt_path, std::string weights_path, int gpu_id = -1);
        Data Forward(int net_id);
        Data Forward(VecMat&& input_image, int net_id);
        Data Forward(VecMat& input_image, int net_id)
        {
            return Forward(std::move(input_image), net_id);
        }

        void SetMemoryDataLayer(std::string layer_name, VecMat&& input_image, int net_id);
        void SetMemoryDataLayer(std::string layer_name, VecMat& input_image, int net_id)
        {
            SetMemoryDataLayer(layer_name, std::move(input_image), net_id);
        }

        void SetDevice(int gpu_id);
        ~CaffeBinding();

    private:
        std::vector<std::shared_ptr<Net<float> > > nets_; //WARNING. MEMORY LEAK WHEN NETS RELEASE RUNTIME
        std::vector<string> prototxts_;
    };
}
