import tensorflow as tf
import numpy as np
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.framework import ops
from rknn_op import *
import time


def euclidean_squared(A, B):
    """
    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.
    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
    jth in second set.
    """

    # tf.ragged FIXME? dangerous, but only used for asserts. Can be omitted.
    shape_A = A.get_shape().as_list()
    shape_B = B.get_shape().as_list()

    # assert (A.dtype == tf.float32 or A.dtype == tf.float64) and (B.dtype == tf.float32 or B.dtype == tf.float64)
    # assert len(shape_A) == 3 and len(shape_B) == 3
    # assert shape_A[0] == shape_B[0]  # and shape_A[1] == shape_B[1]

    # Finds euclidean distance using property (a-b)^2 = a^2 + b^2 - 2ab
    sub_factor = -2 * tf.matmul(A, tf.transpose(B, perm=[0, 2, 1]))  # -2ab term
    dotA = tf.expand_dims(tf.reduce_sum(A * A, axis=2), axis=2)  # a^2 term
    dotB = tf.expand_dims(tf.reduce_sum(B * B, axis=2), axis=1)  # b^2 term
    return tf.abs(sub_factor + dotA + dotB)


class RKNNTest(test.TestCase):

    def test_bare_running(self):
        """
        This test is only for checking if there are no crashes when we run the op. No ragged tensors are used.

        :return:
        """
        n_batch = 2
        n_features = 3
        n_neighbors = 10
        a = np.array([0, 1000, 3000, 3500, 3700, 4800, 5500])
        b = np.random.uniform(size=(5500, 3))
        a = tf.constant(a, dtype=tf.int32)
        b = tf.constant(b, dtype=tf.float32)
        with self.test_session():
            result = rknn_op.RaggedKnn(num_neighbors=int(n_neighbors), row_splits=a, data=b, add_splits=False)
        # result = rknn.RaggedKnn(a,b)

    def test_uniform_test(self):
        num_batch = 32
        num_features = 8
        num_neighbors = 16
        num_vertices_per_batch = 3200
        data = np.random.uniform(size=(num_batch, num_vertices_per_batch, num_features))
        row_splits = np.arange(0, num_vertices_per_batch*(num_batch + 1), num_vertices_per_batch)

        print(row_splits)

        data = tf.constant(data, dtype=tf.float32)
        row_splits = tf.constant(row_splits, dtype=tf.int32)

        t0 = time.time()
        x = euclidean_squared(data, data)
        _, x = tf.nn.top_k(-x, num_neighbors)
        t1 = time.time()
        tf_time =t1-t0

        print(x.numpy())

        data = tf.RaggedTensor.from_row_splits(values=tf.reshape(data, (-1, num_features)), row_splits=row_splits)

        t0 = time.time()
        y,_ = rknn_ragged(data, num_neighbors=num_neighbors)
        t1 = time.time()
        rk_time =t1-t0

        y = tf.reshape(y.values, (num_batch, num_vertices_per_batch, num_neighbors))

        mismatch = float(np.sum(x.numpy()!=y.numpy()))/x.numpy().size


        print(y.numpy())


        print(x.shape)
        print(y.shape)

        print(mismatch)
        print(tf_time, rk_time)
        assert(mismatch < 0.01)


if __name__ == '__main__':
    test.main()
