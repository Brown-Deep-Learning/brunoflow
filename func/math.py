"""
This module contains definitions for autodiff versions of functions in the 'math' module
"""

import numpy as np
from .function import make_function, pointwise_backward
from ..ad import Node

##################### EXPONENTS #####################

sqrt = make_function(
    lambda x: np.sqrt(x),
    pointwise_backward(lambda out, x: 1/(2*out))
)

exp = make_function(
    lambda x: np.exp(x),
    pointwise_backward(lambda out, x: out)
)

log = make_function(
    lambda x: np.log(x),
    pointwise_backward(lambda out, x: 1/x)
)

##################### TRIGOMETRIC #####################

sin = make_function(
    lambda x: np.sin(x),
    pointwise_backward(lambda out, x: np.cos(x))
)

cos = make_function(
    lambda x: np.cos(x),
    pointwise_backward(lambda out, x: -np.sin(x))
)

tan = make_function(
    lambda x: np.tan(x),
    pointwise_backward(lambda out, x: 1 + out*out)
)

asin = make_function(
    lambda x: np.arcsin(x),
    pointwise_backward(lambda out, x: 1/np.sqrt(1 - x*x))
)

acos = make_function(
    lambda x: np.arccos(x),
    pointwise_backward(lambda out, x: -1/np.sqrt(1 - x*x))
)

atan = make_function(
    lambda x: np.arctan(x),
    pointwise_backward(lambda out, x: 1/(1 + x*x))
)

atan2 = make_function(
    lambda a, b: np.arctan2(a, b),
    pointwise_backward(lambda out, a, b: ( b/(a*a + b*b) , -a/(a*a + b*b)  ))
)
Node.__np2bf[np.arctan2] = atan2

##################### HYPERBOLIC #####################

sinh = make_function(
    lambda x: np.sinh(x),
    pointwise_backward(lambda out, x: np.cosh(x))
)

cosh = make_function(
    lambda x: np.cosh(x),
    pointwise_backward(lambda out, x: np.sinh(x))
)

tanh = make_function(
    lambda x: np.tanh(x),
    pointwise_backward(lambda out, x: 1 - out*out)
)

asinh = make_function(
    lambda x: np.arcsinh(x),
    pointwise_backward(lambda out, x: 1/np.sqrt(x*x + 1))
)

acosh = make_function(
    lambda x: np.arccosh(x),
    pointwise_backward(lambda out, x: 1/np.sqrt(x*x - 1))
)

atanh = make_function(
    lambda x: np.arctanh(x),
    pointwise_backward(lambda out, x: 1/(1 - x*x))
)

##################### EXTREMA #####################

def min_deriv(out, a, b):
    mask = (a < b).astype(float)
    return mask, 1 - mask
minimum = make_function(
    lambda a, b: np.minimum(a, b),
    pointwise_backward(min_deriv)
)
Node.__np2bf[np.minimum] = minimum

def max_deriv(out, a, b):
    mask = (a > b).astype(float)
    return mask, 1 - mask
maximum = make_function(
    lambda a, b: np.maximum(a, b),
    pointwise_backward(max_deriv)
)
Node.__np2bf[np.maximum] = maximum

##################### INTEGER CONVERSION #####################

floor = make_function(
    lambda x: np.floor(x),
)

ceil = make_function(
    lambda x: np.ceil(x),
)

round = make_function(
    lambda x: np.round(x),
)

##################### SANITY CHECKS #####################

isfinite = make_function(
    lambda x: np.isfinite(x),
)

isinf = make_function(
    lambda x: np.isinf(x),
)

isnan = make_function(
    lambda x: np.isnan(x),
)

