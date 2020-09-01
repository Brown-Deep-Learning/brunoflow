import unittest as ut
import brunoflow as bf
import numpy as np
import torch
from . import utils


class ShapeTestCase(ut.TestCase):
    pass

######################################################

utils.add_tests(ShapeTestCase,
    basename = 'test_reshape_flatten',
    fn = lambda x: bf.reshape(x, (-1,)),
    torch_fn = lambda x: torch.reshape(x, (-1,)),
    inputs = utils.random_inputs([
        [(4)], [(3,4)], [(5,4,3)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_reshape_unflatten',
    fn = lambda x: bf.reshape(x, (2, -1)),
    torch_fn = lambda x: torch.reshape(x, (2, -1)),
    inputs = utils.random_inputs([
        [(8)], [(4,4)], [(5,4,2)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_squeeze_all',
    fn = lambda x: bf.squeeze(x),
    torch_fn = lambda x: torch.squeeze(x),
    inputs = utils.random_inputs([
        [(4,1)], [(1,4)], [(5,1,3)], [(1,3,1,5)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_expand_dims',
    fn = lambda x: bf.expand_dims(x, 1),
    torch_fn = lambda x: torch.unsqueeze(x, 1),
    inputs = utils.random_inputs([
        [(4)], [(3,4)], [(1,3)], [(3,5,1)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_concat_flat',
    fn = lambda *xs: bf.concat(xs),
    torch_fn = lambda *xs: torch.cat(xs),
    inputs = utils.random_inputs([
        [(4), (4), (4)],
        [(3), (4)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_concat_struct',
    fn = lambda *xs: bf.concat(xs, axis=1),
    torch_fn = lambda *xs: torch.cat(xs, dim=1),
    inputs = utils.random_inputs([
        [(4,2), (4,5), (4,3)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_repeat_flat',
    fn = lambda x: bf.repeat(x, 4, 0),
    torch_fn = lambda x: x.repeat(4),
    inputs = utils.random_inputs([
        [(3)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_repeat_struct',
    fn = lambda x: bf.repeat(x, 4, 1),
    torch_fn = lambda x: x.repeat(1, 4),
    inputs = utils.random_inputs([
        [(3,1)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_stack_0',
    fn = lambda *xs: bf.stack(xs, 0),
    torch_fn = lambda *xs: torch.stack(xs, 0),
    inputs = utils.random_inputs([
        [(3), (3), (3), (3)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_stack_1',
    fn = lambda *xs: bf.stack(xs, 1),
    torch_fn = lambda *xs: torch.stack(xs, 1),
    inputs = utils.random_inputs([
        [(3), (3), (3), (3)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_index_int',
    fn = lambda x: x[1, 1],
    torch_fn = lambda x: x[1, 1],
    inputs = utils.random_inputs([
        [(6,5)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_index_slice',
    fn = lambda x: x[1:3, 1:4],
    torch_fn = lambda x: x[1:3, 1:4],
    inputs = utils.random_inputs([
        [(6,5)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_index_slice/int',
    fn = lambda x: x[1:3, 1],
    torch_fn = lambda x: x[1:3, 1],
    inputs = utils.random_inputs([
        [(6,5)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_index_array',
    fn = lambda x: x[[1, 3, 4]],
    torch_fn = lambda x: x[[1, 3, 4]],
    inputs = utils.random_inputs([
        [(6,5)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_index_array/array',
    fn = lambda x: x[[1, 3, 4], [0, 1, 3]],
    torch_fn = lambda x: x[[1, 3, 4], [0, 1, 3]],
    inputs = utils.random_inputs([
        [(6,5)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_split_size',
    fn = lambda x: bf.split(x, 3, 0),
    torch_fn = lambda x: torch.split(x, 3, 0),
    inputs = utils.random_inputs([
        [(9)],
        [(9,2)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_split_indices',
    fn = lambda x: bf.split(x, [2, 7], 0),
    torch_fn = lambda x: torch.split(x, [2, 5, 2], 0),
    inputs = utils.random_inputs([
        [(9)],
        [(9,2)]
    ])
)

utils.add_tests(ShapeTestCase,
    basename = 'test_unstack',
    fn = lambda x: bf.unstack(x, 0),
    torch_fn = lambda x: torch.unbind(x, 0),
    inputs = utils.random_inputs([
        [(5,3)],
        [(5,2,3)]
    ])
)


