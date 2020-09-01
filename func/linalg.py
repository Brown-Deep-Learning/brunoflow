"""
This module implements autodiff versions of common linear algebra functions.
"""

import numpy as np
from .function import make_function
from .reductions import *
from . import math
from ..ad import Node
    
def transpose(x, axes):
    """
    Permute the axes of a tensor

    Tensorflow equivalent: tf.transpose(x, axes)

    PyTorch equivalent: torch.Tensor.permute(x, axes)
    """
    return __transpose(x, axes)
def transpose_backward(out_val, out_grad, x, axes):
    inverse_perm = np.arange(len(axes))[np.argsort(axes)]
    return np.transpose(out_grad, inverse_perm), None
__transpose = make_function(
    lambda x, axes: np.transpose(x, axes),
    transpose_backward
)

def matrix_transpose(x):
    """
    Permute the last two axes of a tensor

    Tensorflow equivalent: tf.linalg.matrix_transpose(x)

    PyTorch equivalent: torch.transpose(x, -1, -2)
    """
    axes = list(range(x.ndim))
    axes[-2] = x.ndim - 1
    axes[-1] = x.ndim - 2
    return transpose(x, axes)
    
def __np_matrix_transpose(x):
    """
    A numpy version of matrix_transpose, since numpy lacks such a transpose function
        that works on batches of matrices.
    This is used as a helper function later on in this module
    """
    axes = list(range(x.ndim))
    axes[-2] = x.ndim - 1
    axes[-1] = x.ndim - 2
    return np.transpose(x, axes)

def diag(x, k=0):
    """
    Extract the k-th diagonal of a (batch of) matrix

    Tensorflow equivalent: tf.linalg.diag_part(x, k)

    PyTorch equivalent: torch.diagonal(x, k, dim1=-2, dim2=-1),
    """
    return _diag(x, k)
def diag_backward(out_val, out_grad, x, k):
    if x.ndim == 2:
        return diagflat_nonsquare(out_grad, k, x.shape)
    else:
        out_grad = np.reshape(out_grad, (-1, out_grad.shape[-1]))
        b = out_grad.shape[0]
        n = x.shape[-2]
        m = x.shape[-1]
        grad = np.zeros((b, n, m))
        for i in range(b):
            grad[i] = diagflat_nonsquare(out_grad[i], k, grad[i].shape)
        return np.reshape(grad, x.shape), None
def diagflat_nonsquare(diag_vec, k, out_shape):
    diag_mat = np.diagflat(diag_vec, k)
    ds = diag_mat.shape
    os = out_shape
    # Add extra padding rows/cols, if needed
    diag_mat = np.pad(diag_mat, [(0, max(0, os[0]-ds[0])), (0, max(0, os[1]-ds[1]))], 'constant')
    # Delete any extraneous rows/cols, if needed
    diag_mat = diag_mat[0:os[0], 0:os[1]]
    return diag_mat
_diag = make_function(
    lambda x, k: np.diagonal(x, offset=k, axis1=-2, axis2=-1),
    diag_backward
)

def trace(x):
    """
    Compute the trace of a (batch of) matrix

    Tensorflow equivalent: tf.linalg.trace(x)

    PyTorch equivalenet: torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)
    """
    return reduce_sum(diag(x, k=0), axis=-1)

def det(x):
    """
    Compute the determinant of a (batch of) matrix

    Tensorflow equivalent: tf.linalg.det(x)

    PyTorch equivalent: torch.det(x)
    """
    return __det(x)
__det = make_function(
    lambda x: np.linalg.det(x),
    lambda out_val, out_grad, x: np.expand_dims(np.expand_dims(out_val * out_grad, -1), -1) * __np_matrix_transpose(np.linalg.inv(x))
)

def inv(x):
    """
    Compute the inverse of a (batch of) matrix

    Tensorflow equivalent: tf.linalg.inv(x)

    PyTorch equivalent: torch.inverse(x)
    """
    return __inv(x)
def inv_backward(out_val, out_grad, x):
    out_val_T = __np_matrix_transpose(out_val)
    return -np.matmul(out_val_T, np.matmul(out_grad, out_val_T))
__inv = make_function(
    lambda x: np.linalg.inv(x),
    inv_backward
)

def norm(x, axis=None):
    """
    Compute the norm of a (batch of) vector or matrix

    If axis is an int, this function interprets this axis of x as corresponding to a vector,
        and it computes the L2 norm of this vector.

    If axis is a tuple (int, int), this function interprets these two axes of x as
        corresponding to a matrix, and it computes the Frobenius norm of this matrix

    Tensorflow equivalent:
        tf.norm(x, 'euclidean', axis=-1)        [for vector norm]
        tf.norm(x, 'euclidean', axis=[-2, -1])  [for matrix norm]

    PyTorch equivalent:
        torch.norm(x, None, dim=-1)         [for vector norm]
        torch.norm(x, 'fro', dim=(-1, -2))  [for matrix norm]
        
    """
    x_2 = x*x
    if isinstance(axis, tuple):
        assert(len(axis) == 2)
        a0, a1 = axis
        # Convert to absolute indices, if they are negative
        if a0 < 0:
            a0 = x.ndim + a0
        if a1 < 0:
            a1 = x.ndim + a1
        x_red = reduce_sum(x_2, axis=a0)
        if a1 > a0:
            a1 -= 1
        x_red = reduce_sum(x_red, axis=a1)
    else:
        x_red = reduce_sum(x_2, axis=axis)
    return math.sqrt(x_red)

# Matrix multiplication and its infix operator (@)
matmul = make_function(
    lambda A, B: np.matmul(A, B),
    lambda out_val, out_grad, A, B: ( np.matmul(out_grad, __np_matrix_transpose(B)), np.matmul(__np_matrix_transpose(A), out_grad) )
)
Node.__matmul__ = matmul
Node.__rmatmul__ = lambda A, B: Node.__mul__(B, A)