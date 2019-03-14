
"""

Implementation of a Gaussian process (GP).

"""

import tensorflow as tf
from neuralprocesses.utils.tf_utils import define_scope


class GaussianProcess:

    # noinspection PyStatementEffect
    def __init__(self, kernel, name="GaussianProcess"):
        with tf.variable_scope(name):
            self._kernel = kernel

            self.sample

    @define_scope
    def sample(self):
        """
        Computes a sample function from the GP, using a Cholesky decomposition of the kernel, as is described here:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
        :return: Vector of y-values that correspond to the x-values of the kernel.
        """

        b = tf.shape(self._kernel)[0]  # Batch size
        n = tf.shape(self._kernel)[1]  # Number of abscissas

        # Compute the Cholesky decomposition of the kernel (using 64 bit precision for stability)
        cholesky = tf.cast(tf.cholesky(tf.cast(self._kernel, tf.float64)), tf.float32)

        # Get n samples from a (1D) normal distribution
        samples = tf.random_normal((b, n))

        # Taking the product results in a sample from a multivariate normal distribution with the given kernel
        # (correlation matrix). The reshaping is done to flatten a matrix to a vector for each batch-entry.
        y = tf.reshape(
            tf.matmul(cholesky, samples[..., None]),
            (-1, n)
        )

        return y


def squared_exponential_kernel(x, length_scale=1.0, coupling_scale=1.0, noise_scale=0.01,
                               name="squared_exponential_kernel"):

    with tf.variable_scope(name):
        n = tf.shape(x)[1]

        # Evaluate squared exponential function on all pairs of coordinates
        diff_squared = tf.square(tf.expand_dims(x, 1) - tf.expand_dims(x, 2))
        exponent = -0.5 * diff_squared / tf.square(length_scale)
        kernel = tf.square(coupling_scale) * tf.exp(exponent)

        # Add noise to diagonal (improves numerical stability)
        kernel += tf.square(noise_scale) * tf.eye(n)

        return kernel

