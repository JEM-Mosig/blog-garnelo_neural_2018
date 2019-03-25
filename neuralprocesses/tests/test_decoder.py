import unittest

import tensorflow as tf
import numpy as np

from neuralprocesses.np.decoder import MLPDecoder
from neuralprocesses.np.encoder import DeterministicMLPEncoder
from neuralprocesses.np.aggregator import MeanAggregator


class MLPDecoderTEST(unittest.TestCase):
    def test_shapes(self):
        tf.reset_default_graph()
        tf.set_random_seed(2019)

        # Construct a 2-batch of arbitrary target coordinates
        num_target = 4
        x_target = tf.constant([[1, 2, 3, 4], [9, 8, 7, 1]], dtype=tf.float32)

        # Construct a 2-batch of arbitrary representation 3-vectors
        representation = tf.constant([[.3, 2.8, 5.5], [.3, 2.8, 5.5]], dtype=tf.float32)

        # Construct the decoder and the mean and variance tensors as its output
        dec = MLPDecoder([1, 2, 12, 5])  # Layer sizes are arbitrary
        mean, variance = dec(representation, x_target, num_target)

        # Evaluate the graph
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            m_shape, v_shape = session.run([tf.shape(mean), tf.shape(variance)])

        # batch_size          = 2
        # num_target          = 4
        self.assertTrue(np.array_equal(m_shape, [2, num_target]))
        self.assertTrue(np.array_equal(v_shape, [2, num_target]))

    def test_compatibility_with_encoder(self):
        tf.reset_default_graph()
        tf.set_random_seed(2019)

        # Construct a 2-batch of arbitrary context points
        num_context = 4
        x_context = tf.constant([[1, 2, 3, 4], [9, 8, 7, 1]], dtype=tf.float32)
        y_context = tf.constant([[.3, 2.8, 5.5, 1.2], [.3, 2.8, 5.5, 1.2]], dtype=tf.float32)

        # Construct a 2-batch of arbitrary target coordinates
        num_target = 4
        x_target = tf.constant([[4, 12, 0, -3], [-2, -76, 2, 2]], dtype=tf.float32)

        # Define the encoder and aggregator
        enc = DeterministicMLPEncoder([1, 2, 12, 5])    # Layer sizes are arbitrary, except for the last
        agg = MeanAggregator()
        dec = MLPDecoder([1, 2, 12, 5])                 # Layer sizes are arbitrary

        # Construct the graph with the encoder, aggregator, and decoder
        rep_list = enc(x_context, y_context, num_context)
        rep = agg(rep_list)
        mean, variance = dec(rep, x_target, num_target)

        # Evaluate the graph
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            m_shape, v_shape = session.run([tf.shape(mean), tf.shape(variance)])

        # batch_size          = 2
        # num_target          = 4
        self.assertTrue(np.array_equal(m_shape, [2, num_target]))
        self.assertTrue(np.array_equal(v_shape, [2, num_target]))


if __name__ == '__main__':
    unittest.main()
