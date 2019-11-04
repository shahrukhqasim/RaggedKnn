// ragged_knn_kernel.h
#ifndef RAGGED_KNN_KERNEL_H
#define RAGGED_KNN_KERNEL_H

namespace tensorflow {

namespace functor {

template <typename Device>
struct RaggedKnnOpFunctor {
  void operator()(const Device& d, float *d_data, int *d_row_splits,
                         int *d_vector_vertices, float *d_output,
                         int num_neighbors, int num_features, int num_batch, int num_total_vertices);
};

}  // namespace functor

}  // namespace tensorflow

#endif //RAGGED_KNN_KERNEL_H