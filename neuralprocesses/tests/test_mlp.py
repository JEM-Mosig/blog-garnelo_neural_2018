import unittest

import tensorflow as tf
import numpy as np

from neuralprocesses.np.mlp import MultiLayerPerceptron


class MultiLayerPerceptronTEST(unittest.TestCase):
    def test_shapes_common_layer_spec(self):
        tf.reset_default_graph()
        tf.set_random_seed(2019)

        # Construct a 2-batch of arbitrary inputs
        mlp_input = tf.constant([[.3, 2.8, 5.5, 1.2], [.3, 2.8, 5.5, 1.2]], dtype=tf.float32)

        # Construct the mlp with common layer sizes
        mlp = MultiLayerPerceptron([10, 10, 7])
        output = mlp(mlp_input)

        # Evaluate the graph
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            out_shape = session.run(tf.shape(output))

        # batch_size      = 2
        # MLP output size = 7
        self.assertTrue(np.array_equal(out_shape, [2, 7]))

    def test_shapes_no_hidden_layers(self):
        tf.reset_default_graph()
        tf.set_random_seed(2019)

        # Construct a 2-batch of arbitrary inputs
        mlp_input = tf.constant([[.3, 2.8, 5.5, 1.2], [.3, 2.8, 5.5, 1.2]], dtype=tf.float32)

        # Construct the mlp with no hidden layers
        mlp = MultiLayerPerceptron([33])
        output = mlp(mlp_input)

        # Evaluate the graph
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            out_shape = session.run(tf.shape(output))

        # batch_size      = 2
        # MLP output size = 33
        self.assertTrue(np.array_equal(out_shape, [2, 33]))


if __name__ == '__main__':
    unittest.main()
