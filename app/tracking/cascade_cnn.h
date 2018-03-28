// This file is the main function of CascadeCNN.
// A C++ re-implementation for the paper 
// Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li. Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks. IEEE Signal Processing Letters, vol. 23, no. 10, pp. 1499-1503, 2016. 
//
// Code exploited by Feng Wang <feng.wff@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD lisence.
//
// Please cite Zhang's paper in your publications if this code helps your research.

#pragma once

#include <fstream>
#include <thread>

#include <opencv2/opencv.hpp>
#include "tracking/caffe_binding.h"
#include "thread_group.inc.h"
#include "util/bounding_box.h"

#undef assert
#define assert(_Expression) if(!((_Expression)))printf("error: %s %d : %s\n", __FILE__, __LINE__, (#_Expression))


const int kHeightStart = 640;
const int kWidthStart = 480;

const int kMaxNet12Num = 20;

using VVPoint2d = std::vector<std::vector<cv::Point2d> >;
using RectsAndConf= std::vector<std::pair<cv::Rect2d, float> >;

static VVPoint2d DEFAULT_VECTOR;

namespace FaceInception {
    class CascadeCNN
    {
    public:
        CascadeCNN() : scale_decay_(0.707) {}
        CascadeCNN(std::string net12_definition, std::string net12_weights,
                   std::string net12_stitch_definition, std::string net12_stitch_weights,
                   std::string net24_definition, std::string net24_weights,
                   std::string net48_definition, std::string net48_weights,
                   std::string netLoc_definition, std::string netLoc_weights,
                   int gpu_id = -1);

        CascadeCNN(const CascadeCNN&) = delete;
        void operator=(const CascadeCNN&) = delete;

        std::unique_ptr<caffe::CaffeBinding> kCaffeBinding;
        int net12, net12_stitch, net24, net48, netLoc;
        float scale_decay_;
        int input_width, input_height;

        //Only work for small images.
        RectsAndConf getNet12ProposalAcc(Mat& input_image, double min_confidence = 0.6,
                                        double start_scale = 1, bool do_nms = true, double nms_threshold = 0.3);

        RectsAndConf getNet12Proposal(Mat& input_image, double min_confidence = 0.6,
                                                     double start_scale = 1,bool do_nms = true, double nms_threshold = 0.3);

        RectsAndConf getNet24Refined(std::vector<Mat>& sub_images, std::vector<cv::Rect2d>& image_boxes,
                                                    double min_confidence = 0.7,
                                                    bool do_nms = true, double nms_threshold = 0.3,
                                                    int batch_size = 500,
                                                    bool output_points = false, VVPoint2d& points = DEFAULT_VECTOR);

        RectsAndConf getNet48Final(std::vector<Mat>& sub_images, std::vector<cv::Rect2d>& image_boxes,
                                                  double min_confidence = 0.7,
                                                  bool do_nms = true, double nms_threshold = 0.3,
                                                  int batch_size = 500,
                                                  bool output_points = false, VVPoint2d& points = DEFAULT_VECTOR);

        VVPoint2d GetFineLandmark(Mat& input_image, VVPoint2d& coarse_landmarks,
                                                RectsAndConf& face_rects, double width_factor = 0.25);

        RectsAndConf GetDetection(Mat& input_image, double start_scale = 1, double min_confidence = 0.995,
                                             bool do_nms = true, double nms_threshold = 0.7,
                                             bool output_points = false, VVPoint2d& points = DEFAULT_VECTOR);

    };
}
