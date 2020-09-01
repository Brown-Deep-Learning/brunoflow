"""
This module defines functions that alter the shape of a tensor/tensors without changing
    their contents.
"""

import numpy as np
from collections.abc import Iterable 
from .function import make_function
from ..ad import Node

##################### RESHAPING #####################

def reshape(x, newshape):
    """
    Re-organize the values in a tensor into a tensor with a different shape

    Args:
        x: a tensor (Node or numpy.ndarray)
        newshape: a tuple of ints indicating the new shape

    Returns:
        a tensor of shape newshape containing the values from x

    Tensorflow equivalent: tf.reshape

    PyTorch equivalent: torch.reshape
    """
    return __reshape(x, newshape)
__reshape = make_function(
    np.reshape,
    lambda out_val, out_grad, x, newshape: ( np.reshape(out_grad, x.shape), None )
)

def squeeze(x, axis=None):
    """
    Remove 'singleon' dimensions (i.e. dimensions of size 1) from a tensor

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: which singleton axis to remove (if None, remove all singleton axes)

    Returns:
        a tensor with the contents of x, but with singleton axes removed

    Tensorflow equivalent: tf.squeeze

    PyTorch equivalent: torch.squeeze
    """
    return _squeeze(x, axis)
_squeeze = make_function(
    np.squeeze,
    lambda out_val, out_grad, x, axis: ( np.reshape(out_grad, x.shape), None ) 
)

def expand_dims(x, axis):
    """
    Add a 'singleon' dimension (i.e. a dimension of size 1) to a tensor

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: axis at which to add a singleton dimneions

    Returns:
        a tensor with the contents of x, but with a new singleton dimension

    Tensorflow equivalent: tf.expand_dims

    PyTorch equivalent: torch.unsqueeze
    """
    return __expand_dims(x, axis)
__expand_dims = make_function(
    np.expand_dims,
    lambda out_val, out_grad, x, axis: ( np.reshape(out_grad, x.shape), None ) 
)

##################### COMBINING #####################

def concat(xs, axis=0):
    """
    Concatenate multiple tensors into one tensor

    Args:
        xs: a list of tensors (Node or numpy.ndarray). Shapes must be the same, except
            along axis.
        axis: the axis along which to concatenate the tensors

    Returns:
        a single tensor formed from concatenating xs along axis

    Tensorflow equivalent: tf.concat

    PyTorch equivalent: torch.cat
    """
    return __concat(axis, *xs)
def concat_backward(out_val, out_grad, axis, *xs):
    ret_grads = [None]  # no adjoint for the 'axis' argument
    start_idx = 0
    for x in xs:
        slices = tuple(slice(start_idx, start_idx+dim) if i == axis else slice(0, dim) for i,dim in enumerate(x.shape))
        grad = out_grad[slices]
        ret_grads.append(grad)
        start_idx += x.shape[axis]
    return ret_grads
        
__concat = make_function(
    lambda axis, *xs: np.concatenate(xs, axis),
    concat_backward
)

def repeat(x, n, axis):
    """
    Repeat the values in a tensor multiple times to form a larger tensor

    Args:
        x: a tensor (Node or numpy.ndarray)
        n: number of times to repeat x
        axis: the axis along which to repeat x

    Returns:
        a tensor containing the values of x repeated n times along axis

    Tensorflow equivalent: tf.repeat(x, repeats=[n], axis=axis)

    PyTorch equivalent: torch.repeat_interleave(x, n, axis)
    """
    xs = [x for i in range(0, n)]
    return concat(xs, axis)
    
def stack(xs, axis):
    """
    Stack multiple tensors into a single tensor along a new axis

    Args:
        xs: list of tensors (Node or numpy.ndarray), all with the same shape
        axis: the new axis along which to stack xs

    Returns:
        a tensor formed from stacking xs along a new axis

    Tensorflow equivalent: tf.stack

    PyTorch equivalent: torch.stack
    """
    return concat([expand_dims(x, axis) for x in xs], axis)

##################### UN-COMBINING #####################

def get_item(x, arg):
    """
    Index into a tensor to retrieve an element or sub-tensor.

    Args:
        x: a tensor (Node or numpy.ndarray)
        arg: an indexing expression. This could be:
            * An integer index
            * A list of integer indices
            * A slice expression
            * A list of slice expressions
            * A list of both integer indices and slice expressions
            * etc.
            Any indexing expression supported by numpy is supported here.

    Returns:
        Some element / sub-tensor of x

    Tensorflow equivalent: x[arg]

    PyTorch equivalent: x[arg]
    """
    return Node.__getitem__(x, arg)
def getitem_backward(out_val, out_grad, x, arg):
    grad = np.zeros_like(x)
    grad[arg] = out_grad
    return grad, None
Node.__getitem__ = make_function(
    lambda x, arg: x[arg],
    getitem_backward
)

def split(x, indices_or_sections, axis):
    """
    Split a tensor into multiple sub-tensors

    Args:
        x: a tensor (Node or numpy.ndarray)
        indices_or_sections: either:
            * An integer indicating the number of splits to make
            * A list of integers indicating indices where splits should be made
        axis: axis along which to split x

    Returns:
        a list of tensors formed by splitting x according to the args.

    Tensorflow equivalent: tf.split
        * Note that the second argument to tf.split expects *sizes* of splits, not indices where they occur

    PyTorch equivalent: torch.split
        * Note that the second argument to torch.split expects *sizes* of splits, not indices where they occur
    """
    if isinstance(indices_or_sections, int):
        N = indices_or_sections
        size = x.shape[axis]
        if size % N != 0:
            raise ValueError(f'Size of array along axis must be divisible by number of splits; {size} is not divisible by {N}')
        increment = size // N
        indices_or_sections = [i*increment for i in range(1, N)]
    if isinstance(indices_or_sections, Iterable):
        indices = indices_or_sections
        indices.sort()
        if not all([isinstance(i, int) for i in indices]):
            raise TypeError('indices argument to split must contain only ints')
        indices.insert(0, 0)
        indices.append(x.shape[axis])
        splits = []
        for i in range(0, len(indices)-1):
            slices = []
            for j in range(x.ndim):
                start = indices[i] if j == axis else 0
                stop = indices[i+1] if j == axis else x.shape[j]
                slices.append(slice(start, stop))
            splits.append(x[tuple(slices)])
        return splits
    else:
        raise TypeError(f'Argument to split must be either int or Iterable; got {type(indices_or_sections)} instead')

def unstack(x, axis):
    """
    Unpacks a rank N tensor into multiple rank (N-1) tensors along a given axis.
    This is the inverse of the 'stack' function

    Args:
        x: a tensor (Node or numpy.ndarray)
        axis: axis along which to unpack x

    Returns:
        a list of the sub-tensors of x along axis

    Tensorflow equivalent: tf.unstack

    PyTorch equivalent: torch.unbind
    """
    N = x.shape[axis]
    splits = split(x, N, axis)
    return [squeeze(y) for y in splits]

