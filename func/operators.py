"""
This module contains definitions for autodiff versions of functions that are implemented in
    Python as infix operators (e.g. +, -, *, /)

For unary operators (eg. unary -), the relevant overload method is implemented (e.g. __neg__)

For binary operators, two additional steps are taken:

(1) For an operator __[op]__, the 'reverse' operator __r[op]__ is implemented so that the
    correct function is called when the first argument to the operator is a scalar (int | float)
    and the second argument is a Node.

(2) An entry is added to Node.__np2bf so that the correct function is called when the first
    argument to the operator is a numpy.ndarray and the second argument is a Node.
"""

import numpy as np
from .function import make_function, pointwise_backward
from ..ad import Node

def make_scalar_dispatch_op(op):
    def _op(a, b):
        if isinstance(b, float) or isinstance(b, int):
            return op(b, a)
        else:
            return NotImplemented
    return _op

##################### ARITHMETIC #####################

neg = make_function(
    lambda x: -x,
    pointwise_backward(lambda out, x: -1)
)
Node.__neg__ = neg

abs_ = abs
abs = make_function(
    lambda x: abs_(x),
    pointwise_backward(lambda out, x: np.where(x >=0, np.ones_like(x), -np.ones_like(x)))
)
Node.__abs__ = abs

add = make_function(
    lambda a, b: a + b,
    pointwise_backward(lambda out, a, b: (1, 1))
)
Node.__add__ = add
Node.__radd__ = make_scalar_dispatch_op(add)
Node.__np2bf[np.add] = add

sub = make_function(
    lambda a, b: a - b,
    pointwise_backward(lambda out, a, b: (1, -1))
)
Node.__sub__ = sub
Node.__rsub__ = make_scalar_dispatch_op(sub)
Node.__np2bf[np.subtract] = sub

mul = make_function(
    lambda a, b: a * b,
    pointwise_backward(lambda out, a, b: (b, a))
)
Node.__mul__ = mul
Node.__rmul__ = make_scalar_dispatch_op(mul)
Node.__np2bf[np.multiply] = mul

truediv = make_function(
    lambda a, b: a / b,
    pointwise_backward(lambda out, a, b: (1/b, -a/(b*b)))
)
Node.__truediv__ = truediv
Node.__rtruediv__ = make_scalar_dispatch_op(truediv)
Node.__np2bf[np.true_divide] = truediv

floordiv = make_function(
    lambda a, b: a // b
)
Node.__floordiv__ = floordiv
Node.__rfloordiv__ = make_scalar_dispatch_op(floordiv)
Node.__np2bf[np.floor_divide] = floordiv

mod = make_function(
    lambda a, b: a % b
)
Node.__mod__ = mod
Node.__rmod__ = make_scalar_dispatch_op(mod)
Node.__np2bf[np.mod] = mod

pow = make_function(
    lambda a, b: a ** b,
    pointwise_backward(lambda out, a, b: (b * a**(b-1), np.log(a)*out))
)
Node.__pow__ = pow
Node.__rpow__ = make_scalar_dispatch_op(pow)
Node.__np2bf[np.power] = pow

##################### LOGICAL #####################

logical_not = make_function(
    lambda x: ~x
)
Node.__invert__ = logical_not

logical_and = make_function(
    lambda a, b: a & b
)
Node.__and__ = logical_and
Node.__rand__ = make_scalar_dispatch_op(logical_and)
Node.__np2bf[np.logical_and] = logical_and

logical_or = make_function(
    lambda a, b: a | b
)
Node.__or__ = logical_or
Node.__ror__ = make_scalar_dispatch_op(logical_or)
Node.__np2bf[np.logical_or] = logical_or

logical_xor = make_function(
    lambda a, b: a ^ b
)
Node.__xor__ = logical_xor
Node.__rxor__ = make_scalar_dispatch_op(logical_xor)
Node.__np2bf[np.logical_xor] = logical_xor

##################### PREDICATES #####################

eq = make_function(
    lambda a, b: a == b
)
Node.__eq__ = eq
Node.__np2bf[np.equal] = eq

ne = make_function(
    lambda a, b: a != b
)
Node.__ne__ = ne
Node.__np2bf[np.not_equal] = ne

lt = make_function(
    lambda a, b: a < b
)
Node.__lt__ = lt
Node.__np2bf[np.less] = lt

le = make_function(
    lambda a, b: a <= b
)
Node.__le__ = le
Node.__np2bf[np.less_equal] = le

gt = make_function(
    lambda a, b: a > b
)
Node.__gt__ = gt
Node.__np2bf[np.greater] = gt

ge = make_function(
    lambda a, b: a >= b
)
Node.__ge__ = ge
Node.__np2bf[np.greater_equal] = ge