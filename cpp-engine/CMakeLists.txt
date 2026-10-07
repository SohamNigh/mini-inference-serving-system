cmake_minimum_required(VERSION 3.16)
project(inference_engine CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.4
)
FetchContent_MakeAvailable(catch2)

add_library(inference_core
  kernels.cc
  model.cc
)
target_include_directories(inference_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_executable(inference_tests
  tests/kernels_test.cc
  tests/model_test.cc
)
target_link_libraries(inference_tests PRIVATE inference_core Catch2::Catch2WithMain)
add_test(NAME inference_tests COMMAND inference_tests)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
add_library(inference_proto generated/inference.pb.cc generated/inference.grpc.pb.cc)
target_link_libraries(inference_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
target_include_directories(inference_proto PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/generated)
add_executable(inference_engine main.cc)
target_link_libraries(inference_engine PRIVATE inference_core inference_proto)
