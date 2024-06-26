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

#include "yolox_openvino_plugin/yolox_openvino_plugin.hpp"

namespace detector2d_plugins
{

void YoloxOpenVINO::init(const detector2d_parameters::ParamListener & param_listener)
{
  params_ = param_listener.get_params();
  yolo = std::make_shared<yolox_openvino::YoloXOpenVINO>(this->params_.model_path);
  std::cout << "openvino model loaded : " << this->params_.model_path << std::endl;
}

Detection2DArray YoloxOpenVINO::detect(const cv::Mat & image)
{
  auto objects = yolo->inference(image);
  yolox_openvino::utils::draw_objects(image, objects);
  auto boxes = this->objects_to_detection2d_array(objects);
  return boxes;
}

Detection2DArray YoloxOpenVINO::objects_to_detection2d_array(
  const std::vector<yolox_openvino::Object> & objects)
{
  Detection2DArray boxes;
  for (auto obj : objects) {
    vision_msgs::msg::Detection2D detection;
    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = yolox_openvino::COCO_CLASSES[obj.label];
    hypothesis.hypothesis.score = obj.prob;
    detection.results.push_back(hypothesis);

    detection.bbox.center.position.x = obj.rect.x + obj.rect.width / 2;
    detection.bbox.center.position.y = obj.rect.y + obj.rect.height / 2;

    detection.bbox.size_x = obj.rect.width;
    detection.bbox.size_y = obj.rect.height;

    boxes.detections.push_back(detection);
  }
  return boxes;
}
}// namespace detector2d_plugins

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(detector2d_plugins::YoloxOpenVINO, detector2d_base::Detector)