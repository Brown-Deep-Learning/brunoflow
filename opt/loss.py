"""
This module implements common loss functions
"""

from ..func import math
from ..func.shape import reshape
from ..func.reductions import reduce_sum, reduce_mean
from ..func.activations import log_softmax
from .. import ad
import numpy as np

def mse_loss(output, target, reduction='mean'):
    """
    Compute the squared difference between an output and a target value

    Args:
        output: a tensor (Node or numpy.ndarray)
        target: a tensor (Node or numpy.ndarray)
        reduction: how the computed error should be reduced (see below)

    Returns:
        reduce((output - target)^2), where reduce is specified by the 'reduction' arg:
            * reduction='mean' -> reduce(x) = mean(x)
            * reduction='sum'  -> reduce(x) = sum(x)
            * reduction='none' -> reduce(x) = x

    Tensorflow equivalent: tf.compat.v1.losses.mean_squared_error

    PyTorch equivalent: torch.nn.functional.mse_loss
    """
    diff = output - target
    err = diff * diff
    if reduction == 'none':
        return err
    elif reduction == 'mean':
        return reduce_mean(err)
    elif reduction == 'sum':
        return reduce_sum(err)
    else:
        raise ValueError(f'Unsupported reduction type: {reduction}')

def l1_loss(output, target, reduction='mean'):
    """
    Compute the absolute difference between an output and a target value

    Args:
        output: a tensor (Node or numpy.ndarray)
        target: a tensor (Node or numpy.ndarray)
        reduction: how the computed error should be reduced (see below)

    Returns:
        reduce(|output - target|), where reduce is specified by the 'reduction' arg:
            * reduction='mean' -> reduce(x) = mean(x)
            * reduction='sum'  -> reduce(x) = sum(x)
            * reduction='none' -> reduce(x) = x

    Tensorflow equivalent: tf.compat.v1.losses.absolute_difference

    PyTorch equivalent: torch.nn.functional.l1_loss
    """
    err = abs(output - target)
    if reduction == 'none':
        return err
    elif reduction == 'mean':
        return reduce_mean(err)
    elif reduction == 'sum':
        return reduce_sum(err)
    else:
        raise ValueError(f'Unsupported reduction type: {reduction}')

def bce_loss(output, target, reduction='mean'):
    """
    Compute the binary cross entropy between an output and a target value

    Args:
        output: a tensor (Node or numpy.ndarray)
        target: a tensor (Node or numpy.ndarray)
        reduction: how the computed error should be reduced (see below)

    Returns:
        reduce(bce(output, target)), where reduce is specified by the 'reduction' arg:
            * reduction='mean' -> reduce(x) = mean(x)
            * reduction='sum'  -> reduce(x) = sum(x)
            * reduction='none' -> reduce(x) = x

    Tensorflow equivalent: tf.keras.backend.binary_crossentropy

    PyTorch equivalent: torch.nn.functional.binary_cross_entropy
    """
    err = -(target * math.log(output) + (1.0 - target) * math.log(1.0 - output))
    if reduction == 'none':
        return err
    elif reduction == 'mean':
        return reduce_mean(err)
    elif reduction == 'sum':
        return reduce_sum(err)
    else:
        raise ValueError(f'Unsupported reduction type: {reduction}')

def nll_loss(output, target, reduction='mean'):
    """
    Compute the negative log likelihood of a target class under an output log probability
        distribution.

    Args:
        output: a tensor (Node or numpy.ndarray) containing log probabilities
        target: a numpy.ndarray with dtype=int containing class labels
        reduction: how the computed error should be reduced (see below)

    Returns:
        reduce(-output[target]), where reduce is specified by the 'reduction' arg:
            * reduction='mean' -> reduce(x) = mean(x)
            * reduction='sum'  -> reduce(x) = sum(x)
            * reduction='none' -> reduce(x) = x

    Tensorflow equivalent: tf.keras.losses.categorical_crossentropy(tf.one_hot(target), tf.log(output))

    PyTorch equivalent: torch.nn.functional.nll_loss
    """
    target = ad.value(target)
    shape = target.shape
    output = reshape(output, [-1, output.shape[-1]])
    target = np.reshape(target, [-1])
    err = -output[np.arange(target.size), target]
    if reduction == 'none':
        return reshape(err, shape)
    elif reduction == 'mean':
        return reduce_mean(err)
    elif reduction == 'sum':
        return reduce_sum(err)
    else:
        raise ValueError(f'Unsupported reduction type: {reduction}')

def cross_entropy_loss(output, target, reduction='mean'):
    """
    Compute the cross entropy of logits w.r.t. a target class.
    This is just log softmax followed by the NLL loss.

    Args:
        output: a tensor (Node or numpy.ndarray) containing logits
        target: a numpy.ndarray with dtype=int containing class labels
        reduction: how the computed error should be reduced (see below)

    Returns:
        reduce(nll_loss(log_softmax(output), target)), where reduce is specified by the 'reduction' arg:
            * reduction='mean' -> reduce(x) = mean(x)
            * reduction='sum'  -> reduce(x) = sum(x)
            * reduction='none' -> reduce(x) = x

    Tensorflow equivalent: tf.keras.losses.categorical_crossentropy(tf.one_hot(target), output, from_logits=True)

    PyTorch equivalent: torch.nn.functional.cross_entropy
    """
    return nll_loss(log_softmax(output, -1), target, reduction)

