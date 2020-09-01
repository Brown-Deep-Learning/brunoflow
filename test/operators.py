import unittest as ut
import brunoflow as bf
import numpy as np
import torch
from . import utils


class OperatorsTestCase(ut.TestCase):
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

##################### ARITHMETIC #####################

utils.add_tests(OperatorsTestCase,
    basename = 'test_neg',
    fn = lambda x: -x,
    torch_fn = lambda x: -x,
    inputs = utils.random_inputs(unary_input_shapes)
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_abs',
    fn = lambda x: abs(x),
    torch_fn = lambda x: torch.abs(x),
    inputs = utils.random_inputs(unary_input_shapes)
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_add',
    fn = lambda x, y: x + y,
    torch_fn = lambda x, y: x + y,
    inputs = utils.random_inputs(binary_input_shapes)
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_sub',
    fn = lambda x, y: x - y,
    torch_fn = lambda x, y: x - y,
    inputs = utils.random_inputs(binary_input_shapes)
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_mul',
    fn = lambda x, y: x * y,
    torch_fn = lambda x, y: x * y,
    inputs = utils.random_inputs(binary_input_shapes)
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_truediv',
    fn = lambda x, y: x / y,
    torch_fn = lambda x, y: x / y,
    inputs = utils.random_inputs(binary_input_shapes)
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_floordiv',
    fn = lambda x, y: x // y,
    torch_fn = lambda x, y: torch.tensor(x.numpy() // y.numpy()),
    inputs = utils.random_inputs(binary_input_shapes),
    test_backward = False
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_mod',
    fn = lambda x, y: x % y,
    torch_fn = lambda x, y: torch.tensor(x.numpy() % y.numpy()),
    inputs = utils.random_inputs(binary_input_shapes),
    test_backward = False
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_pow',
    fn = lambda x, y: abs(x) ** y,
    torch_fn = lambda x, y: abs(x) ** y,
    inputs = utils.random_inputs(binary_input_shapes)
)

##################### LOGICAL #####################

utils.add_tests(OperatorsTestCase,
    basename = 'test_logical_not',
    fn = lambda x: ~x,
    torch_fn = lambda x: torch.tensor(~x.numpy()),
    inputs = utils.random_inputs(
        unary_input_shapes,
        rand=lambda size: np.random.randint(-5, 5, size=size)
    ),
    test_backward = False
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_logical_and',
    fn = lambda x, y: x & y,
    torch_fn = lambda x, y: torch.tensor(x.numpy() & y.numpy()),
    inputs = utils.random_inputs(
        binary_input_shapes,
        rand=lambda size: np.random.randint(-5, 5, size=size)
    ),
    test_backward = False
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_logical_or',
    fn = lambda x, y: x | y,
    torch_fn = lambda x, y: torch.tensor(x.numpy() | y.numpy()),
    inputs = utils.random_inputs(
        binary_input_shapes,
        rand=lambda size: np.random.randint(-5, 5, size=size)
    ),
    test_backward = False
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_logical_xor',
    fn = lambda x, y: x ^ y,
    torch_fn = lambda x, y: torch.tensor(x.numpy() ^ y.numpy()),
    inputs = utils.random_inputs(
        binary_input_shapes,
        rand=lambda size: np.random.randint(-5, 5, size=size)
    ),
    test_backward = False
)

##################### PREDICATES #####################

utils.add_tests(OperatorsTestCase,
    basename = 'test_eq',
    fn = lambda x, y: x == y,
    torch_fn = lambda x, y: torch.tensor(x.numpy() == y.numpy()),
    inputs = utils.random_inputs(binary_input_shapes),
    test_backward = False
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_ne',
    fn = lambda x, y: x != y,
    torch_fn = lambda x, y: torch.tensor(x.numpy() != y.numpy()),
    inputs = utils.random_inputs(binary_input_shapes),
    test_backward = False
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_lt',
    fn = lambda x, y: x < y,
    torch_fn = lambda x, y: torch.tensor(x.numpy() < y.numpy()),
    inputs = utils.random_inputs(binary_input_shapes),
    test_backward = False
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_le',
    fn = lambda x, y: x <= y,
    torch_fn = lambda x, y: torch.tensor(x.numpy() <= y.numpy()),
    inputs = utils.random_inputs(binary_input_shapes),
    test_backward = False
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_gt',
    fn = lambda x, y: x > y,
    torch_fn = lambda x, y: torch.tensor(x.numpy() > y.numpy()),
    inputs = utils.random_inputs(binary_input_shapes),
    test_backward = False
)

utils.add_tests(OperatorsTestCase,
    basename = 'test_ge',
    fn = lambda x, y: x >= y,
    torch_fn = lambda x, y: torch.tensor(x.numpy() >= y.numpy()),
    inputs = utils.random_inputs(binary_input_shapes),
    test_backward = False
)
