// ragged_knn_kernel.h
#ifndef RAGGED_KNN_KERNEL_H
#define RAGGED_KNN_KERNEL_H

namespace tensorflow {

namespace functor {

template <typename Device, typename dummy>
struct RaggedKnnOpFunctor {
  void operator()(const Device& d, const float *d_data, const int *d_row_splits, int* d_output_indices,
          float *d_output_distances, int num_neighbors, int num_features, int num_batch, int num_total_vertices);
};


}  // namespace functor

}  // namespace tensorflow

#endif //RAGGED_KNN_KERNEL_H