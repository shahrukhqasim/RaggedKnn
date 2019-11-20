#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "ragged_knn_kernel.h"
#include <queue>



namespace tensorflow {
    typedef Eigen::ThreadPoolDevice CPUDevice;
    typedef Eigen::GpuDevice GPUDevice;


    namespace functor {

        // Redefinition
        struct combined {
            int index;
            float distance;
        };
        class combinedcomparator {
        public:
            int operator() (const combined& p1, const combined& p2)
            {
                return p1.distance < p2.distance;
            }
        };

        // CPU specialization null
        template<typename dummy>
        struct RaggedKnnOpFunctor<CPUDevice, dummy> {
            void operator()(const CPUDevice &d, const float *d_data, const int *d_row_splits, int *d_output_indices,
                    float *d_output_distance, int num_neighbors, int num_features, int num_batch, int num_total_vertices, bool add_splits) {
//                assert(false); // We don't have a CPU implementation sorry :(
                printf("Running CPU implementation (in testing)!\n");

                for(int batch = 0; batch < num_batch; batch++) {
                    int num_vertices_in_batch = d_row_splits[batch-1] - d_row_splits[batch];
                    for(int i_vertex_in_batch = 0; i_vertex_in_batch < num_vertices_in_batch; i_vertex_in_batch++) {
                        std::priority_queue <combined, std::vector<combined>, combinedcomparator> topneighbors;

                        for(int j_vertex_in_batch = 0; j_vertex_in_batch < num_vertices_in_batch; j_vertex_in_batch++) {
                            int index = j_vertex_in_batch;
                            float distance = 0;
                            for(int i_feature = 0; i_feature < num_vertices_in_batch; i_feature++) {
                                float a = d_data[i_vertex_in_batch*num_features + i_feature];
                                float b = d_data[j_vertex_in_batch*num_features + i_feature];
                                distance += (a-b)*(a+b);
                            }

                            if (topneighbors.size() < num_neighbors) {
                                topneighbors.push({index, distance});
                            }
                            else if(topneighbors.top().distance > distance) {
                                topneighbors.pop();
                                topneighbors.push({index, distance});
                            }
                        }

                        for(int j_top_neighbor = 0; j_top_neighbor< num_neighbors; j_top_neighbor++) {
                            d_output_distance[i_vertex_in_batch*(num_neighbors+1)-j_top_neighbor-1] = topneighbors.top().distance;
                            d_output_indices[i_vertex_in_batch*(num_neighbors+1)-j_top_neighbor-1] = topneighbors.top().index;
                        }
                    }
                }
            }
        };

        template<typename Device>
        class RaggedKnnOp : public OpKernel {
        public:
            explicit RaggedKnnOp(OpKernelConstruction *context) : OpKernel(context) {
                OP_REQUIRES_OK(context,
                               context->GetAttr("num_neighbors", &num_neighbors));
                OP_REQUIRES_OK(context,
                               context->GetAttr("add_splits", &add_splits));
            }

            void Compute(OpKernelContext *context) override {
                // printf("Not implemented for CPU.\n");
                // assert false;

                const Tensor &row_splits_tensor = context->input(0);
                const Tensor &data_tensor = context->input(1);

                int total_vertices = data_tensor.dim_size(0);
                int num_features = data_tensor.dim_size(1);

//                int total_vertices = data_tensor.NumElements() / num_features;
                int num_batch = row_splits_tensor.NumElements() -1;


                TensorShape outputShape;
                outputShape.AddDim(total_vertices);
                outputShape.AddDim(num_neighbors);

                Tensor *output_tensor_indices = NULL;
                OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor_indices));
                Tensor *output_tensor_distances = NULL;
                OP_REQUIRES_OK(context, context->allocate_output(1, outputShape, &output_tensor_distances));

//                printf("About to do the actual computation\n");

                RaggedKnnOpFunctor<Device, int>()(context->eigen_device<Device>(), data_tensor.flat<float>().data(),
                        row_splits_tensor.flat<int>().data(), output_tensor_indices->flat<int>().data(),
                        output_tensor_distances->flat<float>().data(), num_neighbors, num_features,num_batch,
                        total_vertices, add_splits);


            }

        private:
//            int num_features;
            int num_neighbors;
            bool add_splits;
        };

REGISTER_KERNEL_BUILDER(Name("RaggedKnn").Device(DEVICE_CPU), RaggedKnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct RaggedKnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("RaggedKnn").Device(DEVICE_GPU), RaggedKnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

    }

}
