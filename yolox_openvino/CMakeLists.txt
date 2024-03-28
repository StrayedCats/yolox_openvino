cmake_minimum_required(VERSION 3.8)
project(yolox_openvino)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(InferenceEngine)
find_package(ngraph)

# include
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${InferenceEngine_INCLUDE_DIRS}
  ${ngraph_INCLUDE_DIRS}
)
ament_auto_add_library(${PROJECT_NAME} SHARED src/yolox_openvino.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${InferenceEngine_LIBRARIES} ${ngraph_LIBRARIES})

ament_auto_add_executable(yolox_openvino_example example/yolox_openvino_example.cpp)
target_link_libraries(yolox_openvino_example ${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION lib
)

install(
  TARGETS yolox_openvino_example
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_package()