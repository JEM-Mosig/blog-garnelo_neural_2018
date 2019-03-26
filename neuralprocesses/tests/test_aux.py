import unittest

import tensorflow as tf
import numpy as np
from neuralprocesses.np.aux import DataProvider


class DataProviderTEST(unittest.TestCase):
    def test_constantDistribution(self):
        tf.reset_default_graph()
        tf.set_random_seed(2019)

        # Create a distribution that only returns the constant function f(x) = 0
        class ConstantDistribution:
            def __call__(self, queries):
                _, self._x = queries
                return self.sample

            @property
            def sample(self):
                return self._x * 0.0

        # Construct the computational graph
        distribution = ConstantDistribution()

        # Create the DataProvider and its graph
        data_provider = DataProvider(distribution, batch_size=1, domain=(5, 6))
        data = data_provider(plotting_mode=tf.constant(True))

        # Evaluate the generated context and target points
        with tf.Session() as session:
            xc, yc, xt, yt = session.run([data.queries[0][0], data.queries[0][1], data.queries[1], data.targets])

        # All context coordinates must be within the domain (5, 6) specified above
        self.assertTrue(np.all(xc <= 6.0))
        self.assertTrue(np.all(5.0 <= xc))

        # All target coordinates must be within the domain (5, 6) specified above
        self.assertTrue(np.all(xt <= 6.0))
        self.assertTrue(np.all(5.0 <= xt))

        # All context values must be zero
        self.assertTrue(np.all(yc == 0.0))

        # All target values must be zero
        self.assertTrue(np.all(yt == 0.0))


if __name__ == '__main__':
    unittest.main()
