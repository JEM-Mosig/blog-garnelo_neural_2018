
"""

Implementation of auxiliary functions for NPs.

"""

import tensorflow as tf
from neuralprocesses.utils.tf_utils import define_scope

import collections


RegressionInput = collections.namedtuple("RegressionInput", ("queries", "targets"))


class DataProvider:

    # noinspection PyStatementEffect
    def __init__(self,
                 distribution,
                 batch_size=5,
                 domain=(-1, 1),
                 min_num_context_points=3,
                 max_num_context_points=10,
                 min_num_target_points=3,
                 max_num_target_points=10,
                 target_includes_context=True,
                 name="DataGenerator"
                 ):
        """
        Constructor of a DataProvider. Samples functions from the given distribution and creates a RegressionInput
        object to be used with NPs. An example distribution is `(lambda x: GaussianProcess((None, x), kernel))`, where
        `kernel` is some kernel function.
        :param distribution: The distribution function (lambda).
        :param batch_size: Number of samples to taken from the distribution.
        :param domain: Interval (x_min, x_max) over which to sample coordinates.
        :param min_num_context_points: Minimum number of context points.
        :param max_num_context_points: Maximum number of context points.
        :param min_num_target_points: Minimum number of target points.
        :param max_num_target_points: Maximum number of target points.
        :param target_includes_context: If True, then target points include all context points.
        :param name: Variable scope.
        """

        with tf.variable_scope(name):

            self._distribution = distribution
            self._batch_size = batch_size
            self._min_x = domain[0]
            self._max_x = domain[1]
            self._min_num_context_points = min_num_context_points
            self._max_num_context_points = max_num_context_points
            self._min_num_target_points = min_num_target_points
            self._max_num_target_points = max_num_target_points
            self._target_includes_context = target_includes_context

            self.plotting_mode = tf.placeholder(dtype=tf.bool, name="plotting_mode")

            self.values
            self.num_points
            self.coordinates
            self.data

    @define_scope
    def values(self):
        """
        Gives the values of a function that was sampled from the given distribution.
        :return: [batch_size, num_points]-tensor of function values
        """
        return self._distribution(self.coordinates).sample

    @define_scope
    def num_points(self):
        """
        Generates random integers for the numbers of context and target points, respectively.
        :return: num_points (total), num_context_points, num_target_points
        """

        # Choose the number of context points
        num_context_points = tf.random_uniform(
            shape=(),
            minval=self._min_num_context_points,
            maxval=self._max_num_context_points,
            dtype=tf.int32
        )

        # Choose the number of target points
        num_target_points = tf.random_uniform(
            shape=(),
            minval=self._min_num_target_points,
            maxval=self._max_num_target_points,
            dtype=tf.int32
        )

        return num_context_points + num_target_points, num_context_points, num_target_points

    @define_scope
    def coordinates(self):
        """
        Generates a sample of coordinates.
        :return: [batch_size, num_points]-tensor
        """

        def plot_coordinates():
            # Use many equidistant coordinates for plotting
            delta = (self._max_x - self._min_x) / 100
            x = tf.range(self._min_x, self._max_x, delta, dtype=tf.float32)
            multiply = tf.constant([self._batch_size])
            x = tf.reshape(tf.tile(x, multiply), [multiply[0], tf.shape(x)[0]])
            return x

        def training_coordinates():
            # Use a few uniformly random distributed coordinates for training
            num_points, _, _ = self.num_points
            shape = tf.stack([self._batch_size, num_points])

            # Generate a random set of coordinates in the given domain
            return tf.random_uniform(
                shape=shape,
                minval=self._min_x,
                maxval=self._max_x,
                dtype=tf.float32
            )

        return tf.cond(self.plotting_mode, plot_coordinates, training_coordinates)

    @define_scope
    def data(self):
        """
        Assembles a RegressionInput object from a sample of functions.
        :return: RegressionInput object
        """
        _, num_context_points, num_target_points = self.num_points

        x = self.coordinates
        y = self.values

        # Split the coordinates and values into context and target sets
        x_context, x_target = tf.split(x, [num_context_points, num_target_points], axis=1)
        y_context, y_target = tf.split(y, [num_context_points, num_target_points], axis=1)

        if self._target_includes_context:
            queries = ((x_context, y_context), x)
            targets = y
        else:
            queries = ((x_context, y_context), x_target)
            targets = y_target

        return RegressionInput(queries=queries, targets=targets)
