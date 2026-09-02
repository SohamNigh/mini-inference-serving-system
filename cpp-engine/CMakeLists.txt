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
)
target_include_directories(inference_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_executable(inference_tests
  tests/kernels_test.cc
)
target_link_libraries(inference_tests PRIVATE inference_core Catch2::Catch2WithMain)
add_test(NAME inference_tests COMMAND inference_tests)