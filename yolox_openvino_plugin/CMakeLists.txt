cmake_minimum_required(VERSION 3.8)
project(yolox_openvino_plugin)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(OpenCV REQUIRED)
find_package(InferenceEngine)
find_package(ngraph)

file(GLOB_RECURSE SRCS src/*.cpp)
ament_auto_add_library(yolox_openvino_plugin SHARED
  ${SRCS}
)
target_include_directories(yolox_openvino_plugin PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    ${InferenceEngine_INCLUDE_DIRS}
    ${ngraph_INCLUDE_DIRS}
)
ament_target_dependencies(${PROJECT_NAME} OpenCV)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

pluginlib_export_plugin_description_file(detector2d_base yolox_openvino_plugin.xml)

include_directories(include)
ament_auto_package()