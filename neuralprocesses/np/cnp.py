
"""

Implementation of a conditional neural process (CNP).

"""

import tensorflow as tf
import tensorflow_probability as tfp

from neuralprocesses.np.decoder import DeterministicMLPDecoder
from neuralprocesses.np.encoder import DeterministicMLPEncoder
from neuralprocesses.np.aggregator import MeanAggregator
from neuralprocesses.np.aux import RegressionInput


class ConditionalNeuralProcess:

    def __init__(self, encoder="MLP", aggregator="Mean", decoder="MLP", reuse=None, name="ConditionalNeuralProcess"):
        """
        Create a GaussianProcess graph.
        :param name: Name of the variable scope.
        """

        self._name = name
        self._reuse = reuse

        self._encoder = self._choose_encoder(encoder)
        self._aggregator = self._choose_aggregator(aggregator)
        self._decoder = self._choose_decoder(decoder)

    @staticmethod
    def _choose_encoder(encoder_spec):
        if type(encoder_spec) is str:
            if encoder_spec == "MLP":
                return DeterministicMLPEncoder([64, 64, 64, 64])
            else:
                raise ValueError("Unknown encoder specification")
        else:
            return encoder_spec

    @staticmethod
    def _choose_aggregator(aggregator_spec):
        if type(aggregator_spec) is str:
            if aggregator_spec == "Mean":
                return MeanAggregator()
            else:
                raise ValueError("Unknown aggregator specification")
        else:
            return aggregator_spec

    @staticmethod
    def _choose_decoder(decoder_spec):
        if type(decoder_spec) is str:
            if decoder_spec == "MLP":
                return DeterministicMLPDecoder([64, 64, 64, 64])
            else:
                raise ValueError("Unknown encoder specification")
        else:
            return decoder_spec

    def __call__(self, regression_input: RegressionInput):

        with tf.variable_scope(self._name, reuse=self._reuse):

            # Extract information from regression_input
            (x_context, y_context), x_target = regression_input.queries
            y_target = regression_input.targets
            num_context = regression_input.num_context
            num_target = regression_input.num_target

            # Construct the graph with the encoder, aggregator, and decoder
            representation_list = self._encoder(x_context, y_context, num_context)
            representation = self._aggregator(representation_list)
            mean, variance = self._decoder(representation, x_target, num_target)

            # Loss computation
            with tf.variable_scope("loss"):
                dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=variance)
                log_prob = dist.log_prob(y_target)
                loss = -tf.reduce_mean(log_prob)

            return mean, variance, loss
