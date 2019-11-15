#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("RaggedKnn")
    .Attr("num_neighbors: int")
    .Attr("add_splits: bool")
    .Input("row_splits: int32")
    .Input("data: float32")
    .Output("out_indices: int32")
    .Output("out_vertices: float32");