
"""

Implementation of a Gaussian process (GP).

"""

import tensorflow as tf
from neuralprocesses.utils.tf_utils import define_scope


class GaussianProcess:

    # noinspection PyStatementEffect
    def __init__(self, queries, kernel_function, sort_output=False, name="GaussianProcess"):
        """
        Create a GaussianProcess graph.
        :param queries: Tuple ((x_context, y_context), y_target), where each entry has shape [B, :].
        Can also be (None, y_target), if there are no context points.
        :param kernel_function: Lambda function to generate the kernel matrix.
        :param sort_output: ToDo: Implement automatic sorting
        :param name: Name of the variable scope.
        """

        with tf.variable_scope(name):

            self._sort_output = sort_output     # ToDo: If True, all output is sorted by x-values in ascending order
            self._x_target = queries[1]         # Target point coordinates

            if queries[0] is None:
                # No context points given
                self._x_context = tf.constant([[]], dtype=tf.float32)  # Context point coordinates
                self._y_context = tf.constant([[]], dtype=tf.float32)  # Context point values
                self._x = self._x_target                               # All coordinates
                self._no_context_points = True
            else:
                # There are context points
                self._x_context = queries[0][0]                                 # Context point coordinates
                self._y_context = queries[0][1]                                 # Context point values
                self._x = tf.concat([self._x_context, self._x_target], axis=1)  # All coordinates
                self._no_context_points = False

            with tf.variable_scope("shape_params"):
                b = tf.shape(self._x)[0]     # Batch size
                n = tf.shape(self._x)[1]     # Number of coordinates

            self._prior_kernel = kernel_function(self._x)
            self._prior_mean = tf.zeros(shape=(b, n), dtype=tf.float32, name="default_prior_mean")

            self._mean, self._kernel = self.conditioned_mean_and_kernel

            self.sample
            self.variance
            self.standard_deviation

    @define_scope
    def conditioned_mean_and_kernel(self):
        if self._no_context_points:

            # With no context points, the mean vector and kernel matrix remain unchanged
            return self._prior_mean, self._prior_kernel

        else:

            # If context points are given, use them to condition the mean vector and kernel matrix
            y = self._y_context
            n = tf.shape(y)[1]

            inverse_kernel = tf.matrix_inverse(self._prior_kernel)

            # = inverse_kernel[:, :n, :n]  # [B, n, n] block left-top
            b = inverse_kernel[:, n:, :n]  # [B, r, n] block right-top (= transpose of left-bottom due to symmetry)
            c = inverse_kernel[:, n:, n:]  # [B, r, r] block right-bottom

            kern = tf.matrix_inverse(c)                          # [B, r, r]
            mean = -1 * tf.einsum("bij,bjk,bk->bi", kern, b, y)  # [B, n]

            return mean, kern

    @define_scope
    def sample(self):
        """
        Computes a sample function from the GP, using a Cholesky decomposition of the kernel, as is described here:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
        :return: Vector of y-values that correspond to the x-values of the kernel.
        """

        b = tf.shape(self._kernel)[0]  # Batch size
        n = tf.shape(self._kernel)[1]  # Number of coordinates

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

        # Add the mean-vector to the y-values
        y = y + self._mean

        # Prepend the context values
        if not self._no_context_points:
            y = tf.concat([self._y_context, y], axis=1)

        return y

    @define_scope
    def mean(self):
        """
        Returns the mean for each coordinate
        :return: [len(kernel)]-tensor of variances
        """
        # Prepend the context values
        if self._y_context is not None:
            return tf.concat([self._y_context, self._mean], axis=1)
        else:
            return self._mean

    @define_scope
    def variance(self):
        """
        Returns the variance for each coordinate
        :return: [len(kernel)]-tensor of variances
        """
        var = tf.matrix_diag_part(self._kernel)

        c = tf.shape(self._x_context)[1]  # Number of context points

        # Prepend the context values
        if self._y_context is not None:
            return tf.pad(var, [[0, 0], [c, 0]], constant_values=0)
        else:
            return var

    @define_scope
    def standard_deviation(self):
        """
        Returns the standard deviation for each coordinate
        :return: [len(kernel)]-tensor of standard deviations
        """
        return tf.sqrt(self.variance)

    @define_scope
    def coordinates(self):
        return self._x

    @define_scope
    def kernel_matrix(self):
        return self._kernel


########################################################################################################################
# List of common kernel functions
########################################################################################################################


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
