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

#include "yolox_openvino/yolox_openvino.hpp"

namespace yolox_openvino{
    using namespace InferenceEngine;

    YoloXOpenVINO::YoloXOpenVINO(file_name_t path_to_model, std::string device_name,
                                 float nms_th, float conf_th, std::string model_version,
                                 int num_classes, bool p6)
    :AbcYoloX(nms_th, conf_th, model_version, num_classes, p6),
     device_name_(device_name)
    {
        Core ie;
        network_ = ie.ReadNetwork(path_to_model);

        if (network_.getOutputsInfo().size() != 1)
            throw std::logic_error("Sample supports topologies with 1 output only");
        if (network_.getInputsInfo().size() != 1)
            throw std::logic_error("Sample supports topologies with 1 input only");

        InputInfo::Ptr input_info = network_.getInputsInfo().begin()->second;
        input_name_ = network_.getInputsInfo().begin()->first;

        input_info->setPrecision(Precision::FP32);
        auto input_dims = input_info->getInputData()->getDims();
        this->input_h_ = input_dims[2];
        this->input_w_ = input_dims[3];

        if (network_.getOutputsInfo().empty()) {
            std::cerr << "Network outputs info is empty" << std::endl;
            throw std :: runtime_error( "Network outputs info is empty" );
        }
        DataPtr output_info = network_.getOutputsInfo().begin()->second;
        output_name_ = network_.getOutputsInfo().begin()->first;

        output_info->setPrecision(Precision::FP32);
        executable_network_ = ie.LoadNetwork(network_, device_name_);
        infer_request_ = executable_network_.CreateInferRequest();

        generate_grids_and_stride(this->input_w_, this->input_h_, this->strides_, this->grid_strides_);
    }

    std::vector<yolox_openvino::Object> YoloXOpenVINO::inference(const cv::Mat& frame)
    {
        cv::Mat pr_img = static_resize(frame);
        InferenceEngine::Blob::Ptr imgBlob = infer_request_.GetBlob(input_name_);
        InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(imgBlob);
        if (!mblob)
        {
            THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
                << "but by fact we were not able to cast inputBlob to MemoryBlob";
        }
        auto mblobHolder = mblob->wmap();
        float *blob_data = mblobHolder.as<float *>();
        blobFromImage(pr_img, blob_data);

        infer_request_.Infer();

        const InferenceEngine::Blob::Ptr output_blob = infer_request_.GetBlob(output_name_);
        InferenceEngine::MemoryBlob::CPtr moutput = as<InferenceEngine::MemoryBlob>(output_blob);
        if (!moutput) {
            throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                "but by fact we were not able to cast output to MemoryBlob");
        }

        auto moutputHolder = moutput->rmap();
        const float* net_pred = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type*>();

        float scale = std::min(input_w_ / (frame.cols*1.0), input_h_ / (frame.rows*1.0));
        std::vector<yolox_openvino::Object> objects;
        decode_outputs(net_pred, this->grid_strides_, objects, this->bbox_conf_thresh_, scale, frame.cols, frame.rows);
        return objects;
    }

}
