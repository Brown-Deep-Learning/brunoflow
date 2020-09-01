import unittest as ut
import brunoflow as bf
import numpy as np
import torch
from . import utils


class MathTestCase(ut.TestCase):
    pass

######################################################

unary_input_shapes = [
    [None], [()], [(5)], [(2,2)]
]

binary_input_shapes = [
    [None, None], [(), ()], [(5), (5)], [(2,2), (2,2)],
    # Broadcasting
    [(5), None], [None, (5)], [(5), ()], [(), (5)], [(2,2), (2)], [(2), (2,2)]
]

##################### EXPONENTS #####################

utils.add_tests(MathTestCase,
    basename = 'test_sqrt',
    fn = lambda x: bf.sqrt(abs(x)),
    torch_fn = lambda x: torch.sqrt(abs(x)),
    inputs = utils.random_inputs(unary_input_shapes)
)

utils.add_tests(MathTestCase,
    basename = 'test_exp',
    fn = lambda x: bf.exp(x),
    torch_fn = lambda x: torch.exp(x),
    inputs = utils.random_inputs(unary_input_shapes)
)

utils.add_tests(MathTestCase,
    basename = 'test_log',
    fn = lambda x: bf.log(abs(x)),
    torch_fn = lambda x: torch.log(abs(x)),
    inputs = utils.random_inputs(unary_input_shapes)
)

##################### TRIGOMETRIC #####################

utils.add_tests(MathTestCase,
    basename = 'test_sin',
    fn = lambda x: bf.sin(x),
    torch_fn = lambda x: torch.sin(x),
    inputs = utils.random_inputs(unary_input_shapes)
)

utils.add_tests(MathTestCase,
    basename = 'test_cos',
    fn = lambda x: bf.cos(x),
    torch_fn = lambda x: torch.cos(x),
    inputs = utils.random_inputs(unary_input_shapes)
)

utils.add_tests(MathTestCase,
    basename = 'test_tan',
    fn = lambda x: bf.tan(x),
    torch_fn = lambda x: torch.tan(x),
    inputs = utils.random_inputs(unary_input_shapes)
)

utils.add_tests(MathTestCase,
    basename = 'test_asin',
    fn = lambda x: bf.asin(x),
    torch_fn = lambda x: torch.asin(x),
    inputs = utils.random_inputs(
        unary_input_shapes,
        rand=lambda size: 2*np.random.random(size=size) - 1
    )
)

utils.add_tests(MathTestCase,
    basename = 'test_acos',
    fn = lambda x: bf.acos(x),
    torch_fn = lambda x: torch.acos(x),
    inputs = utils.random_inputs(
        unary_input_shapes,
        rand=lambda size: 2*np.random.random(size=size) - 1
    )
)

utils.add_tests(MathTestCase,
    basename = 'test_atan',
    fn = lambda x: bf.atan(x),
    torch_fn = lambda x: torch.atan(x),
    inputs = utils.random_inputs(
        unary_input_shapes,
        rand=lambda size: 2*np.random.random(size=size) - 1
    )
)

utils.add_tests(MathTestCase,
    basename = 'test_atan2',
    fn = lambda x, y: bf.atan2(x, y),
    torch_fn = lambda x, y: torch.atan2(x, y),
    inputs = utils.random_inputs(
        binary_input_shapes,
        rand=lambda size: 2*np.random.random(size=size) - 1
    )
)

##################### HYPERBOLIC #####################

utils.add_tests(MathTestCase,
    basename = 'test_sinh',
    fn = lambda x: bf.sinh(x),
    torch_fn = lambda x: torch.sinh(x),
    inputs = utils.random_inputs(unary_input_shapes)
)

utils.add_tests(MathTestCase,
    basename = 'test_cosh',
    fn = lambda x: bf.cosh(x),
    torch_fn = lambda x: torch.cosh(x),
    inputs = utils.random_inputs(unary_input_shapes)
)

utils.add_tests(MathTestCase,
    basename = 'test_tanh',
    fn = lambda x: bf.tanh(x),
    torch_fn = lambda x: torch.tanh(x),
    inputs = utils.random_inputs(unary_input_shapes)
)

utils.add_tests(MathTestCase,
    basename = 'test_asinh',
    fn = lambda x: bf.asinh(x),
    torch_fn = lambda x: torch.log(x + torch.sqrt(x*x + 1)),
    inputs = utils.random_inputs(unary_input_shapes)
)

utils.add_tests(MathTestCase,
    basename = 'test_acosh',
    fn = lambda x: bf.acosh(x),
    torch_fn = lambda x: torch.log(x + torch.sqrt(x*x - 1)),
    inputs = utils.random_inputs(
        unary_input_shapes,
        rand = lambda size: 1 + np.random.random(size=size)
    )
)

utils.add_tests(MathTestCase,
    basename = 'test_atanh',
    fn = lambda x: bf.atanh(x),
    torch_fn = lambda x: 0.5 * torch.log((1+x)/(1-x)),
    inputs = utils.random_inputs(
        unary_input_shapes,
        rand=lambda size: 2*np.random.random(size=size) - 1
    )
)

##################### EXTREMA #####################

utils.add_tests(MathTestCase,
    basename = 'test_minimum',
    fn = lambda x, y: bf.minimum(x, y),
    torch_fn = lambda x, y: torch.min(x, y),
    inputs = utils.random_inputs(binary_input_shapes)
)

utils.add_tests(MathTestCase,
    basename = 'test_maximum',
    fn = lambda x, y: bf.maximum(x, y),
    torch_fn = lambda x, y: torch.max(x, y),
    inputs = utils.random_inputs(binary_input_shapes)
)

##################### INTEGER CONVERSION #####################

utils.add_tests(MathTestCase,
    basename = 'test_floor',
    fn = lambda x: bf.floor(x),
    torch_fn = lambda x: torch.tensor(np.floor(x.numpy())),
    inputs = utils.random_inputs(unary_input_shapes),
    test_backward = False
)

utils.add_tests(MathTestCase,
    basename = 'test_ceil',
    fn = lambda x: bf.ceil(x),
    torch_fn = lambda x: torch.tensor(np.ceil(x.numpy())),
    inputs = utils.random_inputs(unary_input_shapes),
    test_backward = False
)

utils.add_tests(MathTestCase,
    basename = 'test_round',
    fn = lambda x: bf.round(x),
    torch_fn = lambda x: torch.tensor(np.round(x.numpy())),
    inputs = utils.random_inputs(unary_input_shapes),
    test_backward = False
)

##################### SANITY CHECKS #####################

utils.add_tests(MathTestCase,
    basename = 'test_isfinite',
    fn = lambda x: bf.isfinite(x),
    torch_fn = lambda x: torch.tensor(np.isfinite(x.numpy())),
    inputs = utils.random_inputs(unary_input_shapes),
    test_backward = False
)

utils.add_tests(MathTestCase,
    basename = 'test_isinf',
    fn = lambda x: bf.isinf(x),
    torch_fn = lambda x: torch.tensor(np.isinf(x.numpy())),
    inputs = utils.random_inputs(unary_input_shapes),
    test_backward = False
)

utils.add_tests(MathTestCase,
    basename = 'test_isnan',
    fn = lambda x: bf.isnan(x),
    torch_fn = lambda x: torch.tensor(np.isnan(x.numpy())),
    inputs = utils.random_inputs(unary_input_shapes),
    test_backward = False
)