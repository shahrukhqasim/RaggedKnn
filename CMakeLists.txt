cmake_minimum_required(VERSION 3.13)
project(ragged_nearest_neighbors)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(ragged_nearest_neighbors
        ragged_knn.cc
        ragged_knn_kernel.cc
        ragged_knn_kernel.cu.cc
        ragged_knn_kernel.h)
