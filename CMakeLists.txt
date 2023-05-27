cmake_minimum_required(VERSION 3.0.0)
project(ffn_project VERSION 0.1.0)

find_package(CUDA 11.3 REQUIRED)

#include(CTest)
#enable_testing()

set(CMAKE_CUDA_COMPILER ${CUDA_ROOT}/bin/nvcc)
enable_language(CUDA)


include_directories(${CUDA_ROOT}/include ${CUDA_ROOT}/include/cuda /usr/include)

link_directories(${CUDA_ROOT}/lib64 /usr/lib/x86_64-linux-gnu)
link_libraries(libcudart.so libcudnn.so libcublas.so) 

add_subdirectory(src)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
