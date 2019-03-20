
"""

This file defines auxiliary constructs (functions/decorators/etc.) that simplify the
implementation of TensorFlow models.

"""

import functools
import tensorflow as tf


def tf_sort_by_col(x, col, descending=False):
    """
    Sorts the tensor `x` by the values in column `col`.
    Code inspired by https://stackoverflow.com/a/41708102/6760298.
    :param x: A (non-ragged) tensor.
    :param col: Index of the column after which rows should be sorted.
    :param descending: If True, values will be in descending order.
    :return: Sorted tensor.
    """
    if descending:
        return tf.gather(x, tf.nn.top_k(x[:, col], k=tf.shape(x)[0]).indices)
    else:
        return tf.gather(x, tf.nn.top_k(-x[:, col], k=tf.shape(x)[0]).indices)


"""

The following code is inspired by 
https://danijar.com/structuring-your-tensorflow-models/

"""


def double_wrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@double_wrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator
