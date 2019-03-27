
"""

Implementation of data-based stochastic processes.

"""

import tensorflow as tf
from neuralprocesses.utils.tf_utils import define_scope

import numpy as np

mnist = tf.keras.datasets.mnist


class MNISTProcess:

    def __init__(self, name="MNISTProcess"):
        """
        Create the graph for a stochastic process, based on MNIST image data.
        :param name: Name of the variable scope.
        """

        self._name = name

        # Load datasets
        (x, y), _ = mnist.load_data()
        # Rescale gray-scale values to range from 0 to 1
        x = x / 255.0

        self._dataset_size, w, h = np.shape(x)
        self._pixel_count = w * h

        # Reshape
        x = np.reshape(x, (self._dataset_size, self._pixel_count))

        self._data = x

    # noinspection PyStatementEffect
    def __call__(self, queries):
        """
        Generate the computational graph for this MNIST image process.
        :param queries: Tuple ((x_context, y_context), y_target), where each entry has shape [B, :].
        Can also be (None, y_target), if there are no context points.
        :return: coordinates, mean, variance, standard_deviation
        """

        with tf.variable_scope(self._name):
            pixel_count = self._pixel_count

            # Target point coordinates should be between 0 and 1. Scale them to be between 0 and 28**2 - 1
            self._x_target = tf.cast(queries[1] * (pixel_count - 1), dtype=tf.int32)

            if queries[0] is not None:
                # There are context points
                raise ValueError("The MNIST process cannot be conditioned.")

            self.sample

    @define_scope
    def sample(self):
        """
        Computes a sample function from the GP, using a Cholesky decomposition of the kernel, as is described here:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
        :return: Vector of y-values that correspond to the x-values of the kernel.
        """
        x_target = self._x_target

        batch_size = tf.shape(x_target)[0]
        num_target_points = tf.shape(x_target)[1]

        data = tf.constant(self._data, dtype=tf.float32, name="MNIST_data")

        # Pick a random sample image for each entry in the batch
        p = tf.random.uniform((batch_size, 1), maxval=self._dataset_size, dtype=tf.int32)

        # Transform [[1, 2, 3], [4, 5, 6]] and [p0, p1] into
        # [[p0, 1], [p0, 2], [p0, 3], [p1, 4], [p1, 5], [p1, 6]]
        # so we use a different image for each set of target coordinates in the batch
        indices = tf.stack(
            [tf.tile(p, [1, num_target_points]), x_target],
            2
        )

        return tf.gather_nd(data, indices=indices)
