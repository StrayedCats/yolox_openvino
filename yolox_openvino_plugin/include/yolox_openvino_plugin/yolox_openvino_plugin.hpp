// Copyright 2024 StrayedCats.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <detector2d_base/detector2d_base.hpp>
#include <detector2d_param/detector2d_param.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <opencv2/opencv.hpp>

#include "yolox_openvino/yolox_openvino.hpp"
#include "yolox_openvino/utils.hpp"
#include "yolox_openvino/coco_names.hpp"

namespace detector2d_plugins
{
typedef vision_msgs::msg::Detection2DArray Detection2DArray;
class YoloxOpenVINO : public detector2d_base::Detector
{
public:
  void init(const detector2d_parameters::ParamListener &) override;
  Detection2DArray detect(const cv::Mat &) override;

private:
  std::shared_ptr<yolox_openvino::YoloXOpenVINO> yolo;

  Detection2DArray objects_to_detection2d_array(const std::vector<yolox_openvino::Object> &);
  detector2d_parameters::Params params_;
};
}