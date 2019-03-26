import unittest

import tensorflow as tf
import numpy as np

from neuralprocesses.np.encoder import DeterministicMLPEncoder
from neuralprocesses.np.aux import DataProvider
from neuralprocesses.utils.gp import GaussianProcess
from neuralprocesses.utils.gp import squared_exponential_kernel


class DeterministicMLPEncoderTEST(unittest.TestCase):
    def test_shapes(self):
        tf.reset_default_graph()
        tf.set_random_seed(2019)

        # Construct a 2-batch of arbitrary context points
        x_context = tf.constant([[1, 2, 3, 4], [9, 8, 7, 1]], dtype=tf.float32)
        y_context = tf.constant([[.3, 2.8, 5.5, 1.2], [.3, 2.8, 5.5, 1.2]], dtype=tf.float32)

        # Construct the encoder and the representation list as its output
        enc = DeterministicMLPEncoder([1, 2, 12, 5])  # Layer sizes are arbitrary, except for the last
        rep = enc(x_context, y_context, 4)

        # Evaluate the graph
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            rep_shape = session.run(tf.shape(rep))

        # batch_size          = 2
        # num_context         = 4
        # encoder output size = 5
        self.assertTrue(np.array_equal(rep_shape, [2, 4, 5]))

    def test_compatibility_with_data_provider(self):
        tf.reset_default_graph()
        tf.set_random_seed(2019)

        distribution = GaussianProcess(squared_exponential_kernel)

        data_provider = DataProvider(distribution, batch_size=3)
        data = data_provider(plotting_mode=tf.constant(False))

        x_context, y_context = data.queries[0]
        num_context = data.num_context

        # Construct the encoder and the representation list as its output
        enc = DeterministicMLPEncoder([1, 2, 12, 5])  # Layer sizes are arbitrary, except for the last
        rep = enc(x_context, y_context, num_context)

        # Evaluate the graph
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            nc, rep_shape = session.run([num_context, tf.shape(rep)])

        # batch_size          = 3
        # num_context         = num_context
        # encoder output size = 5
        self.assertTrue(np.array_equal(rep_shape, [3, nc, 5]))


if __name__ == '__main__':
    unittest.main()
