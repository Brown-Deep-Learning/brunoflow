"""
This module defines functions that perform tensor reductions
"""

import numpy as np
from .function import make_function
from . import math
from .shape import expand_dims

def reduce_min(x, axis=None):
    """
    Compute the minimum value of a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the minimum value of x
        If axis is not None: a tensor with one fewer dimension than x containing the
            minimum values of x along axis

    Tensorflow equivalent: tf.math.reduce_min

    PyTorch equivalent: torch.min
    """
    return _reduce_min(x, axis)
def reduce_min_backward(out_val, out_grad, x, axis=None):
    grad = np.zeros_like(x)
    if axis is None:
        min_index = np.argmin(x)
        min_index = np.unravel_index(min_index, x.shape)
        grad[min_index] = out_grad
    else:
        min_indices = np.argmin(x, axis)
        min_indices = np.expand_dims(min_indices, axis)
        out_grad = np.expand_dims(out_grad, axis)
        np.put_along_axis(grad, min_indices, out_grad, axis)
    return grad, None    
_reduce_min = make_function(
    lambda x, axis: x.min(axis=axis),
    reduce_min_backward
)

def reduce_max(x, axis=None):
    """
    Compute the maximium value of a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the maximum value of x
        If axis is not None: a tensor with one fewer dimension than x containing the
            maximum values of x along axis

    Tensorflow equivalent: tf.math.reduce_max

    PyTorch equivalent: torch.max
    """
    return _reduce_max(x, axis)
def reduce_max_backward(out_val, out_grad, x, axis=None):
    grad = np.zeros_like(x)
    if axis is None:
        min_index = np.argmax(x)
        min_index = np.unravel_index(min_index, x.shape)
        grad[min_index] = out_grad
    else:
        min_indices = np.argmax(x, axis)
        min_indices = np.expand_dims(min_indices, axis)
        out_grad = np.expand_dims(out_grad, axis)
        np.put_along_axis(grad, min_indices, out_grad, axis)
    return grad, None    
_reduce_max = make_function(
    lambda x, axis: x.max(axis=axis),
    reduce_max_backward
)

def reduce_sum(x, axis=None):
    """
    Compute the sum of values in a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the sum of all values in x
        If axis is not None: a tensor with one fewer dimension than x containing the
            sum of the values in x along axis

    Tensorflow equivalent: tf.math.reduce_sum

    PyTorch equivalent: torch.sum
    """
    return _reduce_sum(x, axis)
def reduce_sum_backward(out_val, out_grad, x, axis=None):
    if axis is None:
        return np.full(x.shape, out_grad), None
    else:
        stacks = [out_grad for i in range(x.shape[axis])]
        return np.stack(stacks, axis), None
_reduce_sum = make_function(
    lambda x, axis: np.sum(x, axis=axis),
    reduce_sum_backward
)

def reduce_mean(x, axis=None):
    """
    Compute the mean of values in a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the mean of all values in x
        If axis is not None: a tensor with one fewer dimension than x containing the
            mean of the values in x along axis

    Tensorflow equivalent: tf.math.reduce_mean

    PyTorch equivalent: torch.mean
    """
    n = x.size if axis is None else x.shape[axis]
    return reduce_sum(x, axis) / n

def reduce_var(x, axis=None):
    """
    Compute the variance of values in a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the variance of all values in x
        If axis is not None: a tensor with one fewer dimension than x containing the
            variance of the values in x along axis

    Tensorflow equivalent: tf.math.reduce_variance

    PyTorch equivalent: torch.var
    """
    mean = reduce_mean(x, axis)
    if axis is None:
        diff = x - mean
    else:
        diff = x - expand_dims(mean, axis)
    return reduce_mean(diff*diff, axis)

def reduce_std(x, axis=None):
    """
    Compute the standard deviation of values in a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the standard deviaion of all values in x
        If axis is not None: a tensor with one fewer dimension than x containing the
            standard deviation of the values in x along axis

    Tensorflow equivalent: tf.math.reduce_std

    PyTorch equivalent: torch.std
    """
    return math.sqrt(reduce_var(x, axis))

def reduce_logsumexp(x, axis=None):
    """
    Compute the log of the sum of the exponent of values in a tensor (overall, or along some axis)

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: the axis along which to perform the reduction (if None, reduce over all axes)

    Returns:
        If axis is None: the log-sum-exp of all values in x
        If axis is not None: a tensor with one fewer dimension than x containing the
            log-sum-exp of the values in x along axis

    Tensorflow equivalent: tf.math.reduce_logsumexp

    PyTorch equivalent: torch.logsumexp
    """
    max_val = reduce_max(x, axis)
    if axis is None:
        exps = math.exp(x - max_val)
    else:
        exps = math.exp(x - expand_dims(max_val, axis))
    return math.log(reduce_sum(exps, axis)) + max_val
    # return math.log(reduce_sum(math.exp(x)))