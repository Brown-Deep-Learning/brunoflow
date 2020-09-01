import unittest as ut
import brunoflow as bf
import numpy as np
import torch
from . import utils


class ActivationsTestCase(ut.TestCase):
    pass

######################################################

input_shapes = [
    [(5)], [(2,5)]
]

##################### EXPONENTS #####################

utils.add_tests(ActivationsTestCase,
    basename = 'test_sigmoid',
    fn = lambda x: bf.sigmoid(x),
    torch_fn = lambda x: torch.sigmoid(x),
    inputs = utils.random_inputs(input_shapes)
)

utils.add_tests(ActivationsTestCase,
    basename = 'test_softplus',
    fn = lambda x: bf.softplus(x),
    torch_fn = lambda x: torch.nn.functional.softplus(x),
    inputs = utils.random_inputs(input_shapes)
)

utils.add_tests(ActivationsTestCase,
    basename = 'test_relu',
    fn = lambda x: bf.relu(x),
    torch_fn = lambda x: torch.nn.functional.relu(x),
    inputs = utils.random_inputs(input_shapes)
)

utils.add_tests(ActivationsTestCase,
    basename = 'test_leakyrelu',
    fn = lambda x: bf.leakyrelu(x),
    torch_fn = lambda x: torch.nn.functional.leaky_relu(x),
    inputs = utils.random_inputs(input_shapes)
)

utils.add_tests(ActivationsTestCase,
    basename = 'test_log_softmax',
    fn = lambda x: bf.log_softmax(x, axis=-1),
    torch_fn = lambda x: torch.nn.functional.log_softmax(x, dim=-1),
    inputs = utils.random_inputs(input_shapes)
)

utils.add_tests(ActivationsTestCase,
    basename = 'test_softmax',
    fn = lambda x: bf.softmax(x, axis=-1),
    torch_fn = lambda x: torch.nn.functional.softmax(x, dim=-1),
    inputs = utils.random_inputs(input_shapes)
)