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

#include <vector>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include "core.hpp"

namespace yolox_openvino{
    class YoloXOpenVINO: public AbcYoloX{
        public:
            YoloXOpenVINO(file_name_t path_to_model, std::string device_name="GPU",
                          float nms_th=0.45, float conf_th=0.3, std::string model_version="0.1.1rc0",
                          int num_classes=80, bool p6=false);
            std::vector<Object> inference(const cv::Mat& frame) override;

        private:
            std::string device_name_;
            std::string input_name_;
            std::string output_name_;
            InferenceEngine::CNNNetwork network_;
            InferenceEngine::ExecutableNetwork executable_network_;
            InferenceEngine::InferRequest infer_request_;
    };
}
