import unittest as ut
import brunoflow as bf
import numpy as np
import torch
from . import utils


class ReductionsTestCase(ut.TestCase):
    pass

######################################################

input_shapes_all = [
    [(5)], [(2,5)]
]

input_shapes_some = [
    [(2,5)], [(5,4,3)]
]

######################################################

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_min_all',
    fn = lambda x: bf.reduce_min(x),
    torch_fn = lambda x: torch.min(x),
    inputs = utils.random_inputs(input_shapes_all)
)

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_min_some',
    fn = lambda x: bf.reduce_min(x, axis=1),
    torch_fn = lambda x: torch.min(x, dim=1)[0],
    inputs = utils.random_inputs(input_shapes_some)
)

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_max_all',
    fn = lambda x: bf.reduce_max(x),
    torch_fn = lambda x: torch.max(x),
    inputs = utils.random_inputs(input_shapes_all)
)

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_max_some',
    fn = lambda x: bf.reduce_max(x, axis=1),
    torch_fn = lambda x: torch.max(x, dim=1)[0],
    inputs = utils.random_inputs(input_shapes_some)
)

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_sum_all',
    fn = lambda x: bf.reduce_sum(x),
    torch_fn = lambda x: torch.sum(x),
    inputs = utils.random_inputs(input_shapes_all)
)

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_sum_some',
    fn = lambda x: bf.reduce_sum(x, axis=1),
    torch_fn = lambda x: torch.sum(x, dim=1),
    inputs = utils.random_inputs(input_shapes_some)
)

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_mean_all',
    fn = lambda x: bf.reduce_mean(x),
    torch_fn = lambda x: torch.mean(x),
    inputs = utils.random_inputs(input_shapes_all)
)

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_mean_some',
    fn = lambda x: bf.reduce_mean(x, axis=1),
    torch_fn = lambda x: torch.mean(x, dim=1),
    inputs = utils.random_inputs(input_shapes_some)
)

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_var_all',
    fn = lambda x: bf.reduce_var(x),
    torch_fn = lambda x: torch.var(x, unbiased=False),
    inputs = utils.random_inputs(input_shapes_all)
)

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_var_some',
    fn = lambda x: bf.reduce_var(x, axis=1),
    torch_fn = lambda x: torch.var(x, dim=1, unbiased=False),
    inputs = utils.random_inputs(input_shapes_some)
)

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_std_all',
    fn = lambda x: bf.reduce_std(x),
    torch_fn = lambda x: torch.std(x, unbiased=False),
    inputs = utils.random_inputs(input_shapes_all)
)

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_std_some',
    fn = lambda x: bf.reduce_std(x, axis=1),
    torch_fn = lambda x: torch.std(x, dim=1, unbiased=False),
    inputs = utils.random_inputs(input_shapes_some)
)

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_logsumexp_all',
    fn = lambda x: bf.reduce_logsumexp(x),
    torch_fn = lambda x: torch.logsumexp(x, dim=tuple(range(x.ndim))),
    inputs = utils.random_inputs(input_shapes_all)
)

utils.add_tests(ReductionsTestCase,
    basename = 'test_reduce_logsumexp_some',
    fn = lambda x: bf.reduce_logsumexp(x, axis=1),
    torch_fn = lambda x: torch.logsumexp(x, dim=1),
    inputs = utils.random_inputs(input_shapes_some)
)