"""

Implementation of decoder modules for neural processes.

"""

import tensorflow as tf
from neuralprocesses.np.mlp import MultiLayerPerceptron


class MLPDecoder:

    def __init__(self, neurons_per_layer, reuse=None, name="MLPDecoder"):
        """
        Deterministic multi-layer perceptron encoder for neural processes.
        :param neurons_per_layer: List or iterator of number of neurons in each layer in the MLP. The length of this
        list specifies the depth of the MLP.
        :param name: Variable scope.
        """

        self._name = name
        self._reuse = reuse
        self.regularizer = None

        self._mlp = MultiLayerPerceptron(
            neurons_per_layer + [2],   # There must be two outputs (mean and variance)
            hidden_activation=tf.nn.relu,
            output_activation=None,
            reuse=tf.AUTO_REUSE
        )

    def __call__(self, representation, x_target, num_target):
        """
        Construct the graph.
        :param representation: [batch_size, rep_vec_size] tensor
        :param x_target: [batch_size, num_target] tensor
        :return: [batch_size, num_target, 2] tensor, containing the mean and variance predicted for each target
        coordinate
        """

        with tf.variable_scope(self._name, reuse=self._reuse):

            # We want to pass through the MLP once for each target point in each batch.
            # Thus, we have to transform shapes [batch_size, rep_vec_size] and [batch_size, num_target] into
            # shape [batch_size * num_target, rep_vec_size + 1], i.e. a list of concatenated x/rep lists

            batch_size, dim_x_and_rep = representation.shape.as_list()
            dim_x_and_rep = dim_x_and_rep + 1

            # Stack rep and x to a single [batch_size, num_target, 1 + rep_vec_size] tensor
            mlp_input = tf.concat([
                # Append a `1` to the shape, so we can concat the representation to the coordinate
                tf.expand_dims(x_target, 2),
                # Repeat the representation for each x_target within each batch
                tf.tile(tf.expand_dims(representation, 1), [1, num_target, 1])],
                axis=2
            )

            # Combine batch and target index dimensions to [batch_size * num_target, rep_vec_size + 1]
            mlp_input = tf.reshape(mlp_input, (batch_size * num_target, -1))
            mlp_input.set_shape((None, dim_x_and_rep))

            # Send this through an MLP
            mlp_output = self._mlp(mlp_input)

            # Reshape to one mean/variance pair for each batch and each target point
            mlp_output = tf.reshape(mlp_output, (batch_size, num_target, 2))  # 2 is the output size of the MLP

            # Split into one table for means and one table for variances
            mean, variance = tf.split(mlp_output, 2, axis=-1)

            # Remove final `1` from shape
            mean = tf.reshape(mean, (batch_size, num_target))

            # Remove final `1` from shape and apply `square` to ensure that value is non-negative
            variance = 0.05 + tf.square(tf.reshape(variance, (batch_size, num_target)))  # ToDo: Make 0.05 adjustable

            return mean, variance
