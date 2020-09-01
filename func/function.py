"""
This module contains the code for defining new autodiff functions.
"""

from collections.abc import Iterable 
import numpy as np
from .. import ad

def make_function(forward, backward=None):
    """
    Define a new autodiff function.

    Args:
        forward: The forward pass function.
            Example: forward = lambda x: sin(x)
        backward: The backward pass function. This may be None, if the forward function is
            not differentiable (e.g. comparison operators such as <, >, and =).
            Args:
                out_val: the output value of the forward pass function. Sometimes this is
                    directly used in the backward pass, so not having to re-compute it can
                    be more efficient.
                out_grad: the gradient of the loss w.r.t. the output value of the forward
                    function.
                *input_vals: the input values that were given to the forward pass function.
            Returns:
                The gradient of the loss w.r.t. each input of the forward pass function.
            Example: if forward = lambda x: sin(x), then
                backward = lambda out_val, out_grad, x: out_grad * -cos(x) 
    """

    def backward_wrapper(*args):
        """
        Helper function that makes sure that the return value of backward is always a list
        (So we can iterate over the adjoints returned by it)
        """
        ret = backward(*args)
        if not isinstance(ret, Iterable) or isinstance(ret, np.ndarray):
            ret = [ret]
        return ret

    def autodiff_function(*args):
        """
        The new autodiff function that we have created
        """
        # Extract raw values from any Nodes in the argument list
        in_vals = [ad.value(a) for a in args]
        # Apply the forward function
        out_val = forward(*in_vals)
        # Return a Node which has all the information necessary to perform the backward pass
        return ad.Node(out_val, backward_wrapper if backward else None, args)

    return autodiff_function


def pointwise_backward(pointwise_back_fn):
    """
    For functions that operate pointwise on tensors, the backward function always takes a particular form:
    
    backward = lambda out_val, out_grad, x: out_grad * <some function of out_val and x>

    In this case, we make the job of the function author a little easier by only requiring them to define
        <some function of out_val and x> and taking care of the multiplication by out_grad automatically
    """

    def backward_func(out_val, out_grad, *input_vals):
        # Invoke the pointwise backward function
        local_adjoints = pointwise_back_fn(out_val, *input_vals)
        # Make sure the returned adjoints are a list we can iterate over
        if not isinstance(local_adjoints, Iterable) or isinstance(local_adjoints, np.ndarray):
            local_adjoints  = [local_adjoints]
        # Multiply each returned adjoint by out_grad
        return [ladj * out_grad if (ladj is not None) else None for i,ladj in enumerate(local_adjoints)]
    
    return backward_func


# Map from numpy pointwise functions to Bluenoflow equivalent (used by ufunc_handler below)
# This will be populated by the other sub-modules of this module which define various functions.
ad.Node.__np2bf = {}
def ufunc_handler(self, ufunc, method, *inputs, **kwargs):
    """
    We would like to allow users to call Bluenoflow functions on a mix of Node and numpy.ndarray
        arguments (e.g. the ndarrays can be numerical constants).

    If a binary infix operator (e.g. +) is called on arguments (a, b) of the following types:

    type(a) = numpy.ndarray
    type(b) = Node

    Then according to Python's operator overloading semantics, the type of the first argument
        (i.e. numpy.ndarray) will be used to look up the __add__ method.
    This is not what we want to happen: any time the argument list contains a Node, we want to
        invoke the Bluenoflow version of that operator, so that gradients propagate correctly.

    Fortunately, numpy provides a mechanism for correcting this problem.
    Any Python class can define a __array_ufunc__ method, which specifies what happens whenever
        one of numpy's "universal functions" (or "ufuncs") is called on an argument list
        containing an object of that class (a "universal function" is another name for a
        pointwise function).
    We can overload Node.__array_ufunc__ to instead call the Bluenoflow version of the function,
        which is what this function does.
    """
    # We only recognize ufuncs that have a Bluenoflow equivalent
    if not (ufunc in ad.Node.__np2bf):
        return NotImplemented
    elif method == '__call__':
        bffunc = ad.Node.__np2bf[ufunc]
        return bffunc(*inputs, **kwargs)
    else:
        return NotImplemented
ad.Node.__array_ufunc__ = ufunc_handler