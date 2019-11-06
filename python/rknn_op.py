import tensorflow as tf
import numpy as np



rknn_op = tf.load_op_library('../ragged_knn_kernel.so')





def rknn_ragged(values, num_neighbors):
    """
    Returns a ragged tensor which contains nearest neighbors for each of the vertex.

    :param values: A ragged tensor of rank 3 and dtype float. Second dimension must be ragged. Values among the second
    dimension should not be lower than num_neighbors.
    :param num_neighbors: Number of neighbors to collect for each of the vertex in the batch
    :return: indices and distances as ragged tensors
    """

    assert (len(values.shape)==3)

    num_batch = values.shape[0]
    num_features = values.shape[2]

    row_splits = values.row_splits
    data = values.values

    row_splits = tf.cast(row_splits, tf.int32)

    indices, distances = rknn_op.RaggedKnn(num_batch = int(num_batch), num_features =int(num_features),
                      num_neighbors=int(num_neighbors), row_splits=row_splits, data=data)

    return tf.RaggedTensor.from_row_splits(values=tf.reshape(indices, [-1, num_neighbors]), row_splits=row_splits),\
           tf.RaggedTensor.from_row_splits(values=tf.reshape(distances, [-1, num_neighbors]), row_splits=row_splits)



# def hello():
#     rt = tf.RaggedTensor.from_row_splits(
#         values=[[3,3], [1,2], [4,5], [1,6], [5,1], [9,7], [2,4]],
#         row_splits=[0, 4, 4, 6, 7])
#
#     print(rt.row_splits)
#     print(rt.values)
#
#     rt = tf.RaggedTensor.from_row_splits(
#         values=[3, 1, 4, 1, 5, 9, 2],
#         row_splits=[0, 4, 4, 6, 7])
#     print(rt)
#
#     num_features = 3
#     num_neighbors = 5
#     num_batch = 7
#
#     a = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7]], dtype=np.float)
#     a = a.tolist()*7
#     r = np.array([0, 7, 14, 21, 28, 35, 42, 49], dtype=np.int)
#
#     rt = tf.RaggedTensor.from_row_splits(values=a, row_splits=r)
#     print(rt)
#
#     i, d = rknn_ragged(rt, num_neighbors)
#
#     print(i)
#
#     print(d)
#
#
# if __name__ == '__main__':
#     hello()