//#define GOOGLE_CUDA
//#if GOOGLE_CUDA

#include "ragged_knn_kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cub/cub/cub.cuh"
#include "tensorflow/core/util/gpu_kernel_helper.h"


namespace tensorflow {
namespace functor {


struct combined {
    int index;
    float distance;
};
__device__ unsigned int upper_bound_2(int *a, unsigned int n, int x) {
    unsigned int l = 0;
    unsigned int h = n; // Not n - 1
    while (l < h) {
        unsigned int mid = (l + h) / 2;
        if (x >= a[mid]) {
            l = mid +1;
        } else {
            h = mid;
        }
    }
    return l;
}



__device__ unsigned int upper_bound_2(combined *a, unsigned int n, combined x) {
    unsigned int l = 0;
    unsigned int h = n; // Not n - 1
    while (l < h) {
        unsigned int mid = (l + h) / 2;
        if (x.distance >= a[mid].distance) {
            l = mid +1;
        } else {
            h = mid;
        }
    }
    return l;
}


template <typename T>__device__ void inline cswap(T*& a, T*& b)
{
    T* c(a); a=b; b=c;
}

__global__ void kernel_partial_find(float *d_data, int *d_row_splits,
                         int *d_vector_vertices, float *d_output_x,
                         int num_neighbors, int num_features, int num_batch) {
    using namespace cub;


    combined*d_output = reinterpret_cast<combined*>(d_output_x);


    // 40,960 bytes as cache for collection and reduction
    // TODO: For newer GPUs, increase the size since shared memory could be higher than this
    volatile __shared__ float s_cache_cnr[20480/ sizeof(float)];
//    __shared__ unsigned int s_cache_indices_cnr[4096];

    __shared__ int s_row_splits[1024];

    if (threadIdx.x < num_batch && threadIdx.x < 1024) {
        s_row_splits[threadIdx.x] = d_row_splits[threadIdx.x];
    }

    __syncthreads();



    int my_batch_index = upper_bound_2(s_row_splits, num_batch, blockIdx.x) - 1;
    int my_vertex_index = blockIdx.x - s_row_splits[my_batch_index];


    __shared__ combined t1[256];
    __shared__ combined t2[256];
    __shared__ combined t3[256];

    combined* s_top_neighbors=t1;
    combined* s_neighbors_new=t2;
    combined* s_neighbors_new_2=t3;

    int n_elements_segment = min((int)(20480/ sizeof(float)/num_features), 256);

    __syncthreads();

    int num_neighbors_current = d_vector_vertices[my_batch_index];
    int split_current = d_row_splits[my_batch_index];

    float my_feature = d_data[split_current * num_features + my_vertex_index * num_features + threadIdx.x%num_features];

    __syncthreads();

    int there = 0;


    __syncthreads();

    for (int i_segment = 0; i_segment * n_elements_segment < num_neighbors_current; i_segment++) {
        __syncthreads();

        int num_vertices_this_segment = min(n_elements_segment, num_neighbors_current - i_segment * n_elements_segment);


        __syncthreads();

        // Collection
        int total_elements_to_be_copied = num_vertices_this_segment * num_features;
        for (int i_thread = threadIdx.x; i_thread < total_elements_to_be_copied; i_thread += 256) {
//            int nf = i_thread % num_features;
            float a = d_data[split_current * num_features +
                             i_segment * n_elements_segment * num_features + i_thread];
            s_cache_cnr[i_thread] = (a-my_feature)*(a-my_feature);
        }

        __syncthreads();

        // Reduction
        // You have n_elements_segment * num_features values
        // You have to reduce it to n_elements_segment values
        int m = 1;
        for (int i_reduction_multiplier = 1; i_reduction_multiplier < num_features; i_reduction_multiplier *= 2) {

            int num_threads_per_feature_block = (int) ceil(num_features / (i_reduction_multiplier * 2.0f));
            {

                for (int extendedThreadIdx = threadIdx.x; extendedThreadIdx < n_elements_segment *
                                                                              num_threads_per_feature_block; extendedThreadIdx += 256) {

                    int first_location = (extendedThreadIdx / num_threads_per_feature_block) * num_features +
                                         2 * i_reduction_multiplier *
                                         (extendedThreadIdx % num_threads_per_feature_block);
                    int second_location = first_location + i_reduction_multiplier;


                    if (first_location < num_features * n_elements_segment)
                        if (second_location < num_features * n_elements_segment)
                            if (first_location / num_features == second_location / num_features) {
                                s_cache_cnr[first_location] += s_cache_cnr[second_location];
                            }

                }

            }
            __syncthreads();
            m += 1;
        }




        if (threadIdx.x < num_vertices_this_segment) {
            s_neighbors_new[threadIdx.x].index = n_elements_segment * i_segment + threadIdx.x;
            s_neighbors_new[threadIdx.x].distance = s_cache_cnr[threadIdx.x * num_features];
        } else if (threadIdx.x < 256){
            s_neighbors_new[threadIdx.x].index = 999999;
            s_neighbors_new[threadIdx.x].distance = 1e+037;
        }
        __syncthreads();


        typedef BlockRadixSort<float, 256, 1, int> BlockRadixSortX;
        // Shared memory
        __shared__ typename BlockRadixSortX::TempStorage temp_storage;
        float x[1];
        int y[1];

        x[0] = {s_neighbors_new[threadIdx.x].distance};
        y[0] = {s_neighbors_new[threadIdx.x].index};

        __syncthreads();
        if (threadIdx.x < 256) {
            BlockRadixSortX(temp_storage).Sort(x, y);
        }
        __syncthreads();


        s_neighbors_new[threadIdx.x].distance = x[0];
        s_neighbors_new[threadIdx.x].index = y[0];
        __syncthreads();

        // Binary search and merge
        if (threadIdx.x < num_vertices_this_segment) {
            // Search for yourself
            int mypos = threadIdx.x;
            mypos += upper_bound_2(s_top_neighbors, there, s_neighbors_new[threadIdx.x]);

            if (mypos < num_neighbors) {
                s_neighbors_new_2[mypos] = s_neighbors_new[threadIdx.x];
            }
        }

        if (threadIdx.x < there) {
            int mypos = threadIdx.x + upper_bound_2(s_neighbors_new, num_vertices_this_segment, s_top_neighbors[threadIdx.x]);

            if (mypos < num_neighbors) {
                s_neighbors_new_2[mypos] = s_top_neighbors[threadIdx.x];
            }
        }
        there = num_neighbors;

        __syncthreads();


        cswap(s_top_neighbors, s_neighbors_new_2);

    }


    __syncthreads();

    if (threadIdx.x < num_neighbors) {
        d_output[split_current*num_neighbors + my_vertex_index*num_neighbors+threadIdx.x] = s_top_neighbors[threadIdx.x];
    }
}

typedef Eigen::GpuDevice GPUDevice;
struct RaggedKnnOpFunctor<GPUDevice> {
  void operator()(const GPUDevice& d, float *d_data, int *d_row_splits,
                         int *d_vector_vertices, float *d_output,
                         int num_neighbors, int num_features, int num_batch, int num_total_vertices) {
        kernel_partial_find<<<num_total_vertices, 256>>>(d_data, d_row_splits,
                         d_vector_vertices, d_output, num_neighbors, num_features, num_batch);
  }
};




}
}


//#endif  // GOOGLE_CUDA
