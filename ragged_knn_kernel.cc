#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "ragged_knn_kernel.h"


namespace tensorflow {
    typedef Eigen::ThreadPoolDevice CPUDevice;
    typedef Eigen::GpuDevice GPUDevice;


    namespace functor {
        // CPU specialization null
        template<typename dummy>
        struct RaggedKnnOpFunctor<CPUDevice, dummy> {
            void operator()(const CPUDevice &d, const float *d_data, const int *d_row_splits, int *d_output_indices,
                    float *d_output_distance, int num_neighbors, int num_features, int num_batch, int num_total_vertices) {
//                assert(false); // We don't have a CPU implementation sorry :(
                printf("Running CPU implementation but its nothing here sorry!\n");
            }
        };

        template<typename Device>
        class RaggedKnnOp : public OpKernel {
        public:
            explicit RaggedKnnOp(OpKernelConstruction *context) : OpKernel(context) {
                OP_REQUIRES_OK(context,
                               context->GetAttr("num_batch", &num_batch));
                OP_REQUIRES_OK(context,
                               context->GetAttr("num_features", &num_features));
                OP_REQUIRES_OK(context,
                               context->GetAttr("num_neighbors", &num_neighbors));
            }

            void Compute(OpKernelContext *context) override {
                // printf("Not implemented for CPU.\n");
                // assert false;

                const Tensor &row_splits_tensor = context->input(0);
                const Tensor &data_tensor = context->input(1);

                int total_vertices = data_tensor.NumElements() / num_features;


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
                        total_vertices);


            }

        private:
            int num_batch;
            int num_features;
            int num_neighbors;
        };

REGISTER_KERNEL_BUILDER(Name("RaggedKnn").Device(DEVICE_CPU), RaggedKnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct RaggedKnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("RaggedKnn").Device(DEVICE_GPU), RaggedKnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

    }

}
