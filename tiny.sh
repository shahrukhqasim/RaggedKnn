TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


echo $TF_CFLAGS
echo $TF_LFLAGS


nvcc -std=c++11 -c -o ragged_knn_kernel.cu.o ragged_knn_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I/usr/local/cuda/include/

# g++ -std=c++11 -shared -o ragged_knn.so ragged_knn.cc \
#   ragged_knn_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -I/usr/local/cuda/include/