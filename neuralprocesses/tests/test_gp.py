import unittest

import tensorflow as tf
import numpy as np
from neuralprocesses.utils.gp import squared_exponential_kernel
from neuralprocesses.utils.gp import GaussianProcess


class GPKernelsTEST(unittest.TestCase):

    def test_squared_exp_kernel_1(self):
        """
        Test if the squared-exponential kernel matrix is constructed correctly, when the
        correlation length and amplitude are equal to 1.0 (and noise = 0.0).
        """
        x = tf.constant([[1, 2, 3, 4], [3, 1, 4, 1]], dtype=tf.float32)
        kern = squared_exponential_kernel(x, 1.0, 1.0, 0.0)
        with tf.Session() as session:
            result = session.run(kern)

        test = np.allclose(
            result,
            # Compare to result obtained with Mathematica
            [
                [[1., 0.606531, 0.135335, 0.011109], [0.606531, 1., 0.606531, 0.135335],
                 [0.135335, 0.606531, 1., 0.606531], [0.011109, 0.135335, 0.606531, 1.]],

                [[1., 0.135335, 0.606531, 0.135335], [0.135335, 1., 0.011109, 1.],
                 [0.606531, 0.011109, 1., 0.011109], [0.135335, 1., 0.011109, 1.]]
            ]
        )

        self.assertTrue(test)

    def test_squared_exp_kernel_2(self):
        """
        Test if the squared-exponential kernel matrix is constructed correctly, when the
        correlation length is 3.0 and amplitude is 2.0 (and noise = 0.0).
        """
        x = tf.constant([[1, 2, 3, 4], [3, 1, 4, 1]], dtype=tf.float32)
        kern = squared_exponential_kernel(x, 3.0, 2.0, 0.0)
        with tf.Session() as session:
            result = session.run(kern)

        test = np.allclose(
            result,
            # Compare to result obtained with Mathematica
            [
                [[4., 3.78384, 3.20295, 2.42612], [3.78384, 4., 3.78384, 3.20295],
                 [3.20295, 3.78384, 4., 3.78384], [2.42612, 3.20295, 3.78384, 4.]],

                [[4., 3.20295, 3.78384, 3.20295], [3.20295, 4., 2.42612, 4.],
                 [3.78384, 2.42612, 4., 2.42612], [3.20295, 4., 2.42612, 4.]]
            ]
        )

        self.assertTrue(test)


class GaussianProcessTEST(unittest.TestCase):

    def test_sample_shape(self):
        """
        Test if GaussianProcess.sample returns an array of the correct shape.
        """
        coordinates = tf.constant([[1, 2, 3, 4], [3, 1, 4, 1], [-12, -11, -10, -8]], dtype=tf.float32)
        gp = GaussianProcess((None, coordinates), lambda x: squared_exponential_kernel(x))

        with tf.Session() as session:
            result = session.run(tf.shape(gp.sample))

        self.assertTrue(np.array_equal(result, [3, 4]))


if __name__ == '__main__':
    unittest.main()
