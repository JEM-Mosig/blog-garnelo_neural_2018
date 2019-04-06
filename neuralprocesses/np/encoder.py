"""

Implementation of encoder modules for neural processes.

"""

import tensorflow as tf
from neuralprocesses.np.mlp import MultiLayerPerceptron
from neuralprocesses.np.aggregator import MeanAggregator


class DeterministicMLPEncoder:

    def __init__(self, neurons_per_layer, aggregator="Mean", reuse=None, name="DeterministicMLPEncoder"):
        """
        Deterministic multi-layer perceptron encoder for neural processes.
        :param neurons_per_layer: List or iterator of number of neurons in each layer in the MLP. The length of this
        list specifies the depth of the MLP. The last entry also specifies the size of the representation vector.
        :param name: Variable scope.
        """

        self._name = name
        self._reuse = reuse
        self.regularizer = None
        self._representation_size = neurons_per_layer[-1]

        self._mlp = MultiLayerPerceptron(
            neurons_per_layer,
            hidden_activation=tf.nn.relu,
            output_activation=None,
            reuse=tf.AUTO_REUSE
        )

        self._aggregator = self._choose_aggregator(aggregator)

    @staticmethod
    def _choose_aggregator(aggregator_spec):
        if type(aggregator_spec) is str:
            if aggregator_spec == "Mean":
                return MeanAggregator()
            else:
                raise ValueError("Unknown aggregator specification")
        else:
            return aggregator_spec

    def __call__(self, x_context, y_context, num_context):
        """
        Construct the graph.
        :param x_context: [batch_size, num_context] tensor
        :param y_context: [batch_size, num_context] tensor
        :return: A matrix of representation vectors (shape [batch_size, neurons_per_layer[-1]])
        """

        with tf.variable_scope(self._name, reuse=self._reuse):

            # We want to pass through the MLP once for each context point in each batch.
            # Thus, we have to transform shape [[batch_size, num_context], [batch_size, num_context]] into
            # shape [batch_size * num_context, 2], i.e. a list of x/y pairs

            batch_size, _ = x_context.shape.as_list()

            # Stack x and y to a single [batch_size, num_context, 2] tensor
            mlp_input = tf.stack([x_context, y_context], axis=-1)

            # Combine batch and context index dimensions
            mlp_input = tf.reshape(mlp_input, (batch_size * num_context, -1))
            mlp_input.set_shape((None, 2))  # Necessary for generated data, where num_context is a tensor

            # Send this through an MLP
            representation = self._mlp(mlp_input)

            # Reshape to one representation for each batch and each context point
            representation = tf.reshape(representation, (batch_size, num_context, self._representation_size))

            # Aggregate
            representation = self._aggregator(representation)

            return representation
