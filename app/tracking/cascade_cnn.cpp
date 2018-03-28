#include "cascade_cnn.h"

using namespace cv;
using namespace std;

namespace FaceInception {
    CascadeCNN::CascadeCNN(string net12_definition, string net12_weights, string net12_stitch_definition, string net12_stitch_weights, string net24_definition, string net24_weights, string net48_definition, string net48_weights, string netLoc_definition, string netLoc_weights, int gpu_id) :
        scale_decay_(0.707)
    {
        kCaffeBinding.reset(new caffe::CaffeBinding());
        net12 = kCaffeBinding->AddNet(net12_definition, net12_weights, gpu_id);
        net12_stitch = kCaffeBinding->AddNet(net12_stitch_definition, net12_stitch_weights, gpu_id);
        net24 = kCaffeBinding->AddNet(net24_definition, net24_weights, gpu_id);
        net48 = kCaffeBinding->AddNet(net48_definition, net48_weights, gpu_id);
        netLoc = kCaffeBinding->AddNet(netLoc_definition, netLoc_weights, gpu_id);
    }

    RectsAndConf CascadeCNN::getNet12ProposalAcc(Mat &input_image, double min_confidence, double start_scale, bool do_nms, double nms_threshold)
    {

        int short_side = min(input_image.cols, input_image.rows);
        assert(log(12.0 / start_scale / (double)short_side) / log(scale_decay_) < kMaxNet12Num);
        Mat start_image;
        if (start_scale != 1)
        {
            resize(input_image, start_image, Size(0, 0), start_scale, start_scale);
        }
        else
        {
            start_image = input_image;
        }

        RectsAndConf accumulate_rects;

        //This number(20000) depends on how much GPU memory you have. Please test and modify it to get maximal speed.
        while (start_image.rows * start_image.cols > 20000 && start_image.rows > 12 && start_image.cols > 12)
        {
            auto net12output = kCaffeBinding->Forward({ start_image }, net12);
            if (!(net12output["bounding_box"].size[1] == 1 && net12output["bounding_box"].data[0] == 0))
            {
                RectsAndConf before_nms;
                for (int i = 0; i < net12output["bounding_box"].size[1]; i++) {
                    Rect2d this_rect = Rect2d(net12output["bounding_box"].data[i * 5 + 1] / start_scale,
                            net12output["bounding_box"].data[i * 5] / start_scale,
                            net12output["bounding_box"].data[i * 5 + 3] / start_scale,
                            net12output["bounding_box"].data[i * 5 + 2] / start_scale);

                    before_nms.push_back(make_pair(this_rect, net12output["bounding_box"].data[i * 5 + 4]));
                }
                if (do_nms && before_nms.size() > 1)
                {
                    vector<int> picked = nms_max(before_nms, 0.5);
                    for (auto p : picked)
                    {
                        accumulate_rects.push_back(before_nms[p]);
                    }
                }
                else
                {
                    accumulate_rects.insert(accumulate_rects.end(), before_nms.begin(), before_nms.end());
                }
            }
            start_scale *= scale_decay_;
            resize(start_image, start_image, Size(input_image.cols * start_scale, input_image.rows * start_scale));
        }
        if (start_image.rows > 12 && start_image.cols > 12)
        {
            vector<pair<Rect, double>> location_and_scale;
            Mat big_image = getPyramidStitchingImage2(start_image, location_and_scale);
            Mat stitch_image_x = Mat::zeros((big_image.rows - 12 + 1) / 2 + 1, (big_image.cols - 12 + 1) / 2 + 1, CV_32FC1);
            Mat stitch_image_y = Mat::zeros((big_image.rows - 12 + 1) / 2 + 1, (big_image.cols - 12 + 1) / 2 + 1, CV_32FC1);
            Mat stitch_image_receptive_field = Mat::zeros((big_image.rows - 12 + 1) / 2 + 1, (big_image.cols - 12 + 1) / 2 + 1, CV_32FC1);
            for (auto& ls : location_and_scale)
            {
                Rect rectInOutput = Rect(max(ls.first.x / 2 - 1, 0), max(ls.first.y / 2 - 1, 0),
                                         (ls.first.width - 12 + 2) / 2, (ls.first.height - 12 + 2) / 2);
                stitch_image_x(rectInOutput) = ls.first.y;
                stitch_image_y(rectInOutput) = ls.first.x;
                stitch_image_receptive_field(rectInOutput) = 12.0 / (ls.second*start_scale);
            }

            Mat stitch_image;
            merge(vector<Mat>{ stitch_image_receptive_field, stitch_image_y, stitch_image_x }, stitch_image);
            kCaffeBinding->SetMemoryDataLayer("stitch_data", { stitch_image }, net12_stitch);
            auto net12output = kCaffeBinding->Forward({ big_image }, net12_stitch);

            if (!(net12output["bounding_box"].size[1] == 1 && net12output["bounding_box"].data[0] == 0))
            {
                for (int i = 0; i < net12output["bounding_box"].size[1]; i++)
                {
                    Rect2d this_rect = Rect2d(net12output["bounding_box"].data[i * 5 + 1],
                            net12output["bounding_box"].data[i * 5],
                            net12output["bounding_box"].data[i * 5 + 3],
                            net12output["bounding_box"].data[i * 5 + 2]);

                    accumulate_rects.push_back(make_pair(this_rect, net12output["bounding_box"].data[i * 5 + 4]));
                }
            }
        }

        RectsAndConf result;

        if (do_nms)
        {
            vector<int> picked = nms_max(accumulate_rects, nms_threshold);
            for (auto& p : picked)
            {
                result.push_back(accumulate_rects[p]);
            }
        }
        else
        {
            result = accumulate_rects;
        }

        return result;
    }

    RectsAndConf CascadeCNN::getNet12Proposal(Mat &input_image, double min_confidence, double start_scale, bool do_nms, double nms_threshold)
    {

        int short_side = min(input_image.cols, input_image.rows);
        assert(log(12.0 / start_scale / (double)short_side) / log(scale_decay_) < kMaxNet12Num);
        vector<double> scales;
        double scale = start_scale;
        if (floor(input_image.rows * scale) < 1200 && floor(input_image.cols * scale) < 1200)
        {
            scales.push_back(scale);
        }
        do
        {
            scale *= scale_decay_;
            if (floor(input_image.rows * scale) < 1200 && floor(input_image.cols * scale) < 1200)
            {
                scales.push_back(scale);
            }
        } while (floor(input_image.rows * scale * scale_decay_) >= 12 && floor(input_image.cols * scale * scale_decay_) >= 12);

        vector<RectsAndConf> sub_rects(scales.size());

        for (int s = 0; s < scales.size(); s++)
        {
            Mat small_image;
            resize(input_image, small_image, Size(0, 0), scales[s], scales[s]);
            auto net12output = kCaffeBinding->Forward({ small_image }, net12);
            if (!(net12output["bounding_box"].size[1] == 1 && net12output["bounding_box"].data[0] == 0))
            {
                RectsAndConf before_nms;
                for (int i = 0; i < net12output["bounding_box"].size[1]; i++)
                {
                    Rect2d this_rect = Rect2d(net12output["bounding_box"].data[i * 5 + 1] / scales[s],
                            net12output["bounding_box"].data[i * 5] / scales[s],
                            net12output["bounding_box"].data[i * 5 + 3] / scales[s],
                            net12output["bounding_box"].data[i * 5 + 2] / scales[s]);

                    before_nms.push_back(make_pair(this_rect, net12output["bounding_box"].data[i * 5 + 4]));
                }

                if (do_nms && before_nms.size() > 1)
                {
                    vector<int> picked = nms_max(before_nms, 0.5);
                    for (auto p : picked)
                    {
                        sub_rects[s].push_back(before_nms[p]);
                    }
                }
                else
                {
                    sub_rects[s].insert(sub_rects[s].end(), before_nms.begin(), before_nms.end());
                }
            }

        }

        RectsAndConf accumulate_rects;
        for (int s = 0; s < scales.size(); s++)
        {
            accumulate_rects.insert(accumulate_rects.end(), sub_rects[s].begin(), sub_rects[s].end());
        }
        RectsAndConf result;
        if (do_nms)
        {

            vector<int> picked = nms_max(accumulate_rects, nms_threshold);
            for (auto& p : picked)
            {
                result.push_back(accumulate_rects[p]);
            }
        }
        else
        {
            result = accumulate_rects;
        }

        return result;
    }

    RectsAndConf CascadeCNN::getNet24Refined(vector<Mat> &sub_images, vector<Rect2d> &image_boxes, double min_confidence, bool do_nms, double nms_threshold, int batch_size, bool output_points, VVPoint2d &points)
    {
        int num = sub_images.size();
        if (num == 0)
            return RectsAndConf();
        assert(sub_images[0].cols == 24 && sub_images[0].rows == 24);
        RectsAndConf rect_and_scores;
        vector<vector<Point2d> > allPoints;

        int total_iter = ceil((double)num / (double)batch_size);
        for (int i = 0; i < total_iter; i++)
        {
            int start_pos = i * batch_size;

            if (i == total_iter - 1)
                batch_size = num - (total_iter - 1) * batch_size;

            vector<Mat> net_input = vector<Mat>(sub_images.begin() + start_pos, sub_images.begin() + start_pos + batch_size);
            auto net24output = kCaffeBinding->Forward(net_input, net24);
            for (int j = 0; j < net24output["Prob"].size[0]; j++)
            {
                if (net24output["Prob"].data[j * 2 + 1] > min_confidence)
                {
                    Rect2d this_rect = Rect2d(image_boxes[start_pos + j].x + image_boxes[start_pos + j].width * net24output["conv5-2"].data[j * 4 + 0],
                            image_boxes[start_pos + j].y + image_boxes[start_pos + j].height * net24output["conv5-2"].data[j * 4 + 1],
                            image_boxes[start_pos + j].width + image_boxes[start_pos + j].width * (net24output["conv5-2"].data[j * 4 + 2] - net24output["conv5-2"].data[j * 4 + 0]),
                            image_boxes[start_pos + j].height + image_boxes[start_pos + j].height * (net24output["conv5-2"].data[j * 4 + 3] - net24output["conv5-2"].data[j * 4 + 1]));

                    rect_and_scores.push_back(make_pair(this_rect, net24output["Prob"].data[j * 2 + 1]));
                }
            }
        }
        RectsAndConf result;

        if (do_nms)
        {
            vector<int> picked = nms_max(rect_and_scores, nms_threshold);
            for (auto& p : picked)
            {
                result.push_back(rect_and_scores[p]);
            }
        }
        else
        {
            result = rect_and_scores;
        }

        return result;
    }

    RectsAndConf CascadeCNN::getNet48Final(vector<Mat> &sub_images, vector<Rect2d> &image_boxes, double min_confidence, bool do_nms, double nms_threshold, int batch_size, bool output_points, vector<vector<Point2d> > &points)
    {
        int num = sub_images.size();

        if (num == 0)
            return RectsAndConf();

        assert(sub_images[0].rows == 48 && sub_images[0].cols == 48);
        RectsAndConf rect_and_scores;
        vector<vector<Point2d> > allPoints;

        int total_iter = ceil((double)num / (double)batch_size);
        for (int i = 0; i < total_iter; i++)
        {
            int start_pos = i * batch_size;
            if (i == total_iter - 1)
                batch_size = num - (total_iter - 1) * batch_size;

            vector<Mat> net_input = vector<Mat>(sub_images.begin() + start_pos, sub_images.begin() + start_pos + batch_size);

            auto net48output = kCaffeBinding->Forward(net_input, net48);
            for (int j = 0; j < net48output["Prob"].size[0]; j++)
            {
                if (net48output["Prob"].data[j * 2 + 1] > min_confidence)
                {
                    Rect2d this_rect = Rect2d(image_boxes[start_pos + j].x + image_boxes[start_pos + j].width * net48output["conv6-2"].data[j * 4 + 0],
                            image_boxes[start_pos + j].y + image_boxes[start_pos + j].height * net48output["conv6-2"].data[j * 4 + 1],
                            image_boxes[start_pos + j].width + image_boxes[start_pos + j].width * (net48output["conv6-2"].data[j * 4 + 2] - net48output["conv6-2"].data[j * 4 + 0]),
                            image_boxes[start_pos + j].height + image_boxes[start_pos + j].height * (net48output["conv6-2"].data[j * 4 + 3] - net48output["conv6-2"].data[j * 4 + 1]));

                    rect_and_scores.push_back(make_pair(this_rect, net48output["Prob"].data[j * 2 + 1]));

                    if (output_points)
                    {
                        vector<Point2d> point_list;
                        for (int p = 0; p < 5; p++)
                        {
                            point_list.push_back(Point2d(net48output["conv6-3"].data[j * 10 + p] * image_boxes[start_pos + j].width + image_boxes[start_pos + j].x,
                                    net48output["conv6-3"].data[j * 10 + p + 5] * image_boxes[start_pos + j].height + image_boxes[start_pos + j].y));
                        }
                        allPoints.push_back(point_list);
                    }
                }
            }
        }

        if (output_points)
            assert(allPoints.size() == rect_and_scores.size());

        RectsAndConf result;
        if (do_nms)
        {
            vector<int> picked = nms_max(rect_and_scores, nms_threshold, IoU_MIN);
            for (auto& p : picked)
            {
                result.push_back(rect_and_scores[p]);

                if (output_points)
                    points.push_back(allPoints[p]);
            }
        }
        else
        {
            result = rect_and_scores;
        }

        return result;
    }

    VVPoint2d CascadeCNN::GetFineLandmark(Mat &input_image, VVPoint2d &coarse_landmarks, RectsAndConf &face_rects, double width_factor)
    {
        vector<Mat> sub_images;
        int face_num = face_rects.size();
        for (int n = 0; n < face_num; n++)
        {
            Mat concated_local_patch = Mat(24, 24, CV_8UC(15));
            vector<Mat> local_patches;
            double width = max(face_rects[n].first.width, face_rects[n].first.height) * width_factor;
            vector<int> from_to;
            from_to.reserve(30);
            for (int p = 0; p < 5; p++)
            {
                Mat local_patch_p = cropImage(input_image,
                                              Rect2d(coarse_landmarks[n][p].x - width / 2,
                                                     coarse_landmarks[n][p].y - width / 2,
                                                     width, width),
                                              Size(24, 24), INTER_LINEAR, BORDER_CONSTANT, DEFAULT_SCALAR);
                local_patch_p = local_patch_p.t();
                local_patches.push_back(local_patch_p);
                from_to.insert(from_to.end(), { p * 3 + 0,p * 3 + 2,
                                                p * 3 + 1,p * 3 + 1,
                                                p * 3 + 2,p * 3 + 0 });
            }

            mixChannels(local_patches, { concated_local_patch }, from_to);
            sub_images.push_back(concated_local_patch);
        }

        auto netLocOutput = kCaffeBinding->Forward(sub_images, netLoc);
        for (int n = 0; n < face_num; n++)
        {
            double width = max(face_rects[n].first.width, face_rects[n].first.height) * width_factor;
            for (int p = 0; p < 5; p++)
            {
                coarse_landmarks[n][p].x = coarse_landmarks[n][p].x - width / 2 + netLocOutput["fc5_" + to_string(p+1)].data[2 * n + 0] * width;
                coarse_landmarks[n][p].y = coarse_landmarks[n][p].y - width / 2 + netLocOutput["fc5_" + to_string(p+1)].data[2 * n + 1] * width;
            }
        }

        return coarse_landmarks;
    }

    RectsAndConf CascadeCNN::GetDetection(Mat &input_image, double start_scale, double min_confidence, bool do_nms, double nms_threshold, bool output_points, vector<vector<Point2d> > &points)
    {
        Mat clone_image = input_image.clone();//for drawing
        auto proposal = getNet12ProposalAcc(clone_image, 0.6, start_scale, do_nms, nms_threshold);

        if (proposal.size() == 0)
            return RectsAndConf();

        vector<Mat> sub_images;
        sub_images.reserve(proposal.size());
        vector<Rect2d> image_boxes;
        image_boxes.reserve(proposal.size());
        for (auto& p : proposal)
        {
            make_rect_square(p.first);

            if (p.first.width < 9 || p.first.height < 9)
                continue;

            Mat sub_image = cropImage(input_image, p.first, Size(24, 24), INTER_LINEAR, BORDER_CONSTANT, DEFAULT_SCALAR);
            sub_images.push_back(sub_image);
            image_boxes.push_back(p.first);
        }
        auto refined = getNet24Refined(sub_images, image_boxes, 0.7, do_nms, nms_threshold, 500);

        if (refined.size() == 0)
            return RectsAndConf();

        vector<Mat> sub_images48;
        sub_images48.reserve(refined.size());
        vector<Rect2d> image_boxes48;
        image_boxes48.reserve(refined.size());

        for (auto& p : refined)
        {
            make_rect_square(p.first);

            if (p.first.width < 9 || p.first.height < 9)
                continue;

            Mat sub_image = cropImage(input_image, p.first, Size(48, 48), INTER_LINEAR, BORDER_CONSTANT, DEFAULT_SCALAR);
            sub_images48.push_back(sub_image);
            image_boxes48.push_back(p.first);
        }

        auto final = getNet48Final(sub_images48, image_boxes48, min_confidence, do_nms, nms_threshold, 500, output_points, points);
        if (output_points && final.size() > 0)
        {
            GetFineLandmark(input_image, points, final);
        }

        return final;
    }

}
