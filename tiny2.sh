TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


echo $TF_CFLAGS
echo $TF_LFLAGS


nvcc -std=c++11 -c -o ragged_knn_kernel.cu.o ragged_knn_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr \
    -L/usr/local/cuda/lib64 -lcudart ${TF_LFLAGS[@]} -DNDEBUG

echo "Step 1"


g++ -std=c++11 -c -o ragged_knn_kernel.o ragged_knn_kernel.cc \
   ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -fPIC \
   -I/usr/local/cuda/include

echo "Step 2"

g++ -std=c++11 -c -o ragged_knn.o ragged_knn.cc \
   ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -fPIC \
   -I/usr/local/cuda/include

echo "Step 3"

g++ -std=c++11 -shared -o ragged_knn_kernel.so ragged_knn_kernel.cu.o ragged_knn_kernel.o ragged_knn.o \
   ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -fPIC \
   -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart ${TF_LFLAGS[@]}

