
"""

Aggregator(s) for the NP

"""

import tensorflow as tf


class MeanAggregator:

    def __init__(self, name="MeanAggregator", reuse=None):
        self._name = name
        self._reuse = reuse

    def __call__(self, representation_list):
        """
        Construct the graph.
        :param representation_list: A matrix of representation vectors (shape [batch_size, num_context, rep_vec_size])
        :return: [batch_size, rep_vec_size] tensor, averaged over all context points.
        """
        with tf.variable_scope(self._name, reuse=self._reuse):
            return tf.reduce_mean(representation_list, 1)
