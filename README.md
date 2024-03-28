# yolox_openvino

```bash
ros2 run detector2d_node detector2d_node_exec --ros-args -p load_target_plugin:=detector2d_plugins::YoloxOpenVINO -p model_path:=${HOME}/yolox_tiny.onnx
```