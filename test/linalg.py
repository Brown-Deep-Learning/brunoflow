import unittest as ut
import brunoflow as bf
import numpy as np
import torch
from . import utils


class LinalgTestCase(ut.TestCase):
    pass

######################################################

utils.add_tests(LinalgTestCase,
    basename = 'test_transpose',
    fn = lambda x: bf.transpose(x, [1, 2, 0]),
    torch_fn = lambda x: x.permute(1, 2, 0),
    inputs = utils.random_inputs([ [(5,4,3)] ])
)

utils.add_tests(LinalgTestCase,
    basename = 'test_matrix_transpose',
    fn = lambda x: bf.matrix_transpose(x),
    torch_fn = lambda x: torch.transpose(x, -2, -1),
    inputs = utils.random_inputs([
        [(4,4)], [(5,4)],
        [(3,5,4)]  # Batch of matrices
    ])
)

utils.add_tests(LinalgTestCase,
    basename = 'test_diag_k=0',
    fn = lambda x: bf.diag(x),
    torch_fn = lambda x: torch.diagonal(x, dim1=-2, dim2=-1),
    inputs = utils.random_inputs([
        [(4,4)], [(5,4)], [(4,5)],
        [(3,5,4)]  # Batch of matrices
    ])
)

utils.add_tests(LinalgTestCase,
    basename = 'test_diag_k=1',
    fn = lambda x: bf.diag(x, k=1),
    torch_fn = lambda x: torch.diagonal(x, offset=1, dim1=-2, dim2=-1),
    inputs = utils.random_inputs([
        [(4,4)], [(5,4)], [(4,5)],
        [(3,5,4)]  # Batch of matrices
    ])
)

utils.add_tests(LinalgTestCase,
    basename = 'test_diag_k=-1',
    fn = lambda x: bf.diag(x, k=-1),
    torch_fn = lambda x: torch.diagonal(x, offset=-1, dim1=-2, dim2=-1),
    inputs = utils.random_inputs([
        [(4,4)], [(5,4)], [(4,5)],
        [(3,5,4)]  # Batch of matrices
    ])
)

utils.add_tests(LinalgTestCase,
    basename = 'test_trace',
    fn = lambda x: bf.trace(x),
    torch_fn = lambda x: torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1),
    inputs = utils.random_inputs([
        [(4,4)], [(5,4)], [(4,5)],
        [(3,5,4)]  # Batch of matrices
    ])
)

utils.add_tests(LinalgTestCase,
    basename = 'test_det',
    fn = lambda x: bf.det(x),
    torch_fn = lambda x: torch.det(x),
    inputs = utils.random_inputs([
        [(4,4)],
        [(3,5,5)]  # Batch of matrices
    ])
)

utils.add_tests(LinalgTestCase,
    basename = 'test_inv',
    fn = lambda x: bf.inv(x),
    torch_fn = lambda x: torch.inverse(x),
    inputs = utils.random_inputs([
        [(4,4)],
        [(3,5,5)]  # Batch of matrices
    ])
)

utils.add_tests(LinalgTestCase,
    basename = 'test_norm_all',
    fn = lambda x: bf.norm(x),
    torch_fn = lambda x: torch.norm(x),
    inputs = utils.random_inputs([
        [(5)], [(4,4)],
    ])
)

utils.add_tests(LinalgTestCase,
    basename = 'test_norm_axis_vector',
    fn = lambda x: bf.norm(x, axis=-1),
    torch_fn = lambda x: torch.norm(x, dim=-1),
    inputs = utils.random_inputs([
        [(3,5)]  # Batch of vectors
    ])
)

utils.add_tests(LinalgTestCase,
    basename = 'test_norm_axis_matrix',
    fn = lambda x: bf.norm(x, axis=(-2,-1)),
    torch_fn = lambda x: torch.norm(x, dim=(-2,-1)),
    inputs = utils.random_inputs([
        [(3,5,5)]  # Batch of matrices
    ])
)

utils.add_tests(LinalgTestCase,
    basename = 'test_matmul',
    fn = lambda x, y: bf.matmul(x, y),
    torch_fn = lambda x, y: torch.matmul(x, y),
    inputs = utils.random_inputs([
        [(4,4), (4,4)], [(3,4), (4,5)],
        [(2,4,4), (2,4,4)], [(2,3,4), (2,4,5)],
        [(2,4,4), (4,4)], [(3,4), (2,4,5)]
    ])
)

utils.add_tests(LinalgTestCase,
    basename = 'test_matmul_op',
    fn = lambda x, y: x @ y,
    torch_fn = lambda x, y: x @ y,
    inputs = utils.random_inputs([
        [(4,4), (4,4)], [(3,4), (4,5)],
        [(2,4,4), (2,4,4)], [(2,3,4), (2,4,5)],
        [(2,4,4), (4,4)], [(3,4), (2,4,5)]
    ])
)