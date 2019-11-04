#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "ragged_knn_kernel.h"



namespace tensorflow {

namespace functor {
// CPU specialization null
    struct TimeTwoFunctor<CPUDevice> {
        void operator()(const CPUDevice &d, int size, const T *in, T *out) {
        }

        void operator()(const CPUDevice &d, float *d_data, int *d_row_splits,
                        int *d_vector_vertices, float *d_output,
                        int num_neighbors, int num_features, int num_batch, int num_total_vertices) {
            assert(false); // We don't have a CPU implementation sorry :(
        }
    };
}

template <typename Device>
class RaggedKnnOp : public OpKernel {
    explicit RaggedKnnOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                   context->GetAttr("num_batch", &num_batch));
        OP_REQUIRES_OK(context,
                   context->GetAttr("num_features", &num_features));
        OP_REQUIRES_OK(context,
                   context->GetAttr("num_neighbors", &num_neighbors));
    }
    void Compute(OpKernelContext* context) override {
        // printf("Not implemented for CPU.\n");
        // assert false;

        const Tensor& vertices_tensor = context->input(0);
        const Tensor& row_splits_tensor = context->input(1);
        const Tensor& data_tensor = context->input(2);

        int total_vertices = data_tensor.NumElements();

        
        TensorShape outputShape;
        outputShape.AddDim(total_vertices*num_neighbors*2);

        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));

        RaggedKnnOpFunctor<Device>()(context->eigen_device<Device>(), data_tensor.flat<float>().data(),
                        row_splits_tensor.flat<int>().data(), vertices_tensor.flat<int>().data(), num_neighbors,
                        num_features,num_batch, total_vertices);


    }
private:
    int num_batch;
    int num_features;
    int num_neighbors;
};


REGISTER_KERNEL_BUILDER(Name("RaggedKnn").Device(DEVICE_CPU), RaggedKnnOp<CPUDevice>);
//#ifdef GOOGLE_CUDA
extern template struct RaggedKnnOpFunctor<GPUDevice>;
REGISTER_KERNEL_BUILDER(Name("RaggedKnn").Device(DEVICE_GPU), RaggedKnnOp<GPUDevice>);

//#endif  // GOOGLE_CUDA


}