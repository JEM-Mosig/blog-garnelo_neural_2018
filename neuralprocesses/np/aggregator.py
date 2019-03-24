
"""

Aggregator(s) for the NP

"""

import tensorflow as tf


class MeanAggregator:

    def __init__(self):
        pass

    def __call__(self, representation_list):
        return tf.reduce_mean(representation_list, -1)
