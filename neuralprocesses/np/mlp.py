
"""

Generic multilayer perceptron (MLP)

"""

import tensorflow as tf


class MultiLayerPerceptron:

    def __init__(self, neurons_per_layer, hidden_activation=tf.nn.relu, output_activation=None,
                 reuse=None, name="MultiLayerPerceptron"):
        """
        Multilayer perceptron.
        :param neurons_per_layer: List or iterator of number of neurons in each layer. Length of this list specifies the
        depth of the MLP
        :param hidden_activation: Activation function for all hidden layers.
        :param output_activation: Activation function for output layer.
        :param name: Variable scope.
        """

        self._name = name
        self._reuse = reuse
        self._num_neurons = neurons_per_layer
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation
        self.regularizer = None

    def __call__(self, input_activation):

        with tf.variable_scope(self._name, reuse=self._reuse):

            # Create hidden layers
            previous = input_activation
            for d, num_neurons in enumerate(self._num_neurons[:-1]):
                previous = tf.layers.dense(
                    previous, num_neurons,
                    activation=self._hidden_activation,
                    kernel_regularizer=self.regularizer,
                    name="dense-" + str(d)
                )

            # Create output layer
            output = tf.layers.dense(
                previous, self._num_neurons[-1],
                activation=self._output_activation,
                kernel_regularizer=self.regularizer,
                name="output"
            )

            return output
