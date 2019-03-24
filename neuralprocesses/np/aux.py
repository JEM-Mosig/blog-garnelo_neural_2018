
"""

Implementation of auxiliary functions for NPs.

"""

import tensorflow as tf
from neuralprocesses.utils.tf_utils import define_scope

import collections


RegressionInput = collections.namedtuple("RegressionInput", ("queries", "targets", "num_context"))


class DataProvider:

    # noinspection PyStatementEffect
    def __init__(self,
                 batch_size=5,
                 domain=(-1, 1),
                 min_num_context_points=3,
                 max_num_context_points=10,
                 min_num_target_points=3,
                 max_num_target_points=10,
                 target_includes_context=True,
                 plotting_mode=False,
                 reuse=None,
                 name="DataProvider"
                 ):
        """
        Constructor of a DataProvider. Samples functions from the given distribution and creates a RegressionInput
        object to be used with NPs. An example distribution is `(lambda x: GaussianProcess((None, x), kernel))`, where
        `kernel` is some kernel function.
        :param batch_size: Number of samples to taken from the distribution.
        :param domain: Interval (x_min, x_max) over which to sample coordinates.
        :param min_num_context_points: Minimum number of context points.
        :param max_num_context_points: Maximum number of context points.
        :param min_num_target_points: Minimum number of target points.
        :param max_num_target_points: Maximum number of target points.
        :param target_includes_context: If True, then target points include all context points.
        :param name: Variable scope.
        """

        self._name = name
        self._reuse = reuse
        self._batch_size = batch_size
        self._min_x = domain[0]
        self._max_x = domain[1]
        self._min_num_context_points = min_num_context_points
        self._max_num_context_points = max_num_context_points
        self._min_num_target_points = min_num_target_points
        self._max_num_target_points = max_num_target_points
        self._target_includes_context = target_includes_context

        self.plotting_mode = plotting_mode

    # noinspection PyStatementEffect
    def __call__(self, distribution, *args, **kwargs):
        """
        Generate the computational graph.
        :param distribution: The distribution function (lambda).
        :return:
        """

        with tf.variable_scope(self._name, reuse=self._reuse):

            # Create the computational graph that generates the distribution and feed it with self.coordinates
            distribution((None, self.coordinates))

            # Values are samples from that distribution
            self.values = distribution.sample

            # Generate parts of the graph that are not returned by `.__call__`
            self.num_points
            self.coordinates

            # Return the data output
            return self.data

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

        return tf.cond(tf.constant(self.plotting_mode), plot_coordinates, training_coordinates)

    @define_scope
    def data(self):
        """
        Assembles a RegressionInput object from a sample of functions.
        :return: RegressionInput object
        """

        x = self.coordinates
        y = self.values

        def plot_data():  # This branch is taken if we are in plotting mode

            # Context points are the same as target points, but randomly shuffled
            r = tf.random_shuffle(tf.range(tf.shape(x)[1]))
            x_context = tf.gather(x, r, axis=1)
            y_context = tf.gather(y, r, axis=1)

            return ((x_context, y_context), x), y, 100  # ToDo: Make 100 a named constant

        def training_data():  # This branch is taken if we are not plotting

            _, num_context_points, num_target_points = self.num_points

            # Split the coordinates and values into context and target sets
            x_context, x_target = tf.split(x, [num_context_points, num_target_points], axis=1)
            y_context, y_target = tf.split(y, [num_context_points, num_target_points], axis=1)

            if self._target_includes_context:
                q = ((x_context, y_context), x)
                t = y
            else:
                q = ((x_context, y_context), x_target)
                t = y_target

            return q, t, num_context_points

        queries, targets, num_context = tf.cond(tf.constant(self.plotting_mode), plot_data, training_data)

        return RegressionInput(queries=queries, targets=targets, num_context=num_context)
