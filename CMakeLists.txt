cmake_minimum_required(VERSION 2.8)
project(ragged_nearest_neighbors)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(ragged_nearest_neighbors
        ragged_knn.cc
        ragged_knn_kernel.cc
        ragged_knn_kernel.cu.cc
        ragged_knn_kernel.h)


execute_process(COMMAND python -c 'import tensorflow as tf' OUTPUT_VARIABLE TF_CFLAGS)
#execute_process(COMMAND python -c "import tensorflow as tf; print\(\" \".join(tf.sysconfig.get_link_flags\(\)\)\)" OUTPUT_VARIABLE TF_LFLAGS)

MESSAGE( ${TF_CFLAGS} )
#MESSAGE(${TF_LFLAGS})
