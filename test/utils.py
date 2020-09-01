from collections.abc import Iterable
import brunoflow as bf
import numpy as np
import random
import torch
from functools import reduce

rtol = 1e-05
atol = 1e-08

class TestInput:
    def __init__(self, name, vals):
        self.name = name
        self.vals = vals

def inputs_to_params(inp, lift_every_other=False):
    ret = []
    for i,val in enumerate(inp.vals):
        if isinstance(val, int) or (isinstance(val, np.ndarray) and val.dtype == int):
            ret.append(val)
        elif not lift_every_other or i % 2 == 0:
            ret.append(bf.Parameter(val))
        else:
            ret.append(val)
    return ret

def inputs_to_torch(inp, requires_grad=False):
    ret = []
    for val in inp.vals:
        if isinstance(val, int):
            ret.append(val)
        elif isinstance(val, float):
            ret.append(torch.tensor([val], requires_grad=requires_grad))
        elif isinstance(val, np.ndarray) and val.dtype == int:
            ret.append(torch.tensor(val))
        else:
            ret.append(torch.tensor(val, requires_grad=requires_grad))
    return ret

def check(self, x, target):
    if isinstance(x, bf.ad.Node):
        x = x.val
    if isinstance(x, np.ndarray):
        if x.dtype == bool or x.dtype == int:
            if not np.array_equal(x, target.numpy()):
                self.assertIs(x, target.numpy())
        elif x.dtype == float:
            if not np.allclose(x, target.numpy(), rtol=rtol, atol=atol):
                self.assertIs(x, target.numpy())
    elif isinstance(x, bool):
        self.assertEqual(x, target.item())
    elif isinstance(x, float):
        if not np.allclose(x, target.item(), rtol=rtol, atol=atol):
            self.assertIs(x, target.item())
    elif isinstance(x, Iterable):
        self.assertEqual(len(x), len(target))
        for i in range(len(x)):
            check(self, x[i], target[i])
    else:
        raise TypeError(f'Unknown type of x in check: {type(x)}')

def add_forward_tests(clss, basename, fn, torch_fn, inputs, lift_every_other=False):
    for inp in inputs:
        def test_forward(self):
            # print('======= FORWARD TEST =======')
            x = inputs_to_params(inp, lift_every_other)
            x_t = inputs_to_torch(inp)
            # print('x:', x)
            # print('x_t:', x_t)
            y = fn(*x)
            y_t = torch_fn(*x_t)
            # print('y:', y)
            # print('y_t:', y_t)
            check(self, y, y_t)
        setattr(clss, f'test_forward_{basename}_{inp.name}', test_forward)

def add_backward_tests(clss, basename, fn, torch_fn, inputs, lift_every_other=False):
    for inp in inputs:
        def test_backward(self):
            x = inputs_to_params(inp, lift_every_other)
            x_t = inputs_to_torch(inp, requires_grad=True)
            y = fn(*x)
            y_t = torch_fn(*x_t)
            if not isinstance(y, Iterable):
                y = [y]
                y_t = [y_t]
            for i in range(len(y)):
                y[i].backprop()
                y_t[i].backward(torch.ones_like(y_t[i]), retain_graph=True)
                grads_x = [x_.grad for x_ in x if isinstance(x_, bf.Parameter)]
                grads_xt = [x_t[i].grad for i in range(len(x_t)) if isinstance(x[i], bf.Parameter)]
                check(self, grads_x, grads_xt)
                y[i].zero_gradients()
                # I *think* this is how you're supposed to do it...
                for x_ in x_t:
                    if x_.requires_grad:
                        x_.grad.zero_()
        setattr(clss, f'test_backward_{basename}_{inp.name}', test_backward)

def add_tests(clss, basename, fn, torch_fn, inputs, test_backward=True):
    
    add_forward_tests(clss, basename, fn, torch_fn, inputs)
    if test_backward:
        add_backward_tests(clss, basename, fn, torch_fn, inputs)
    # Version where some inputs aren't lifted to be Parameters
    if len(inputs[0].vals) > 1:
        add_forward_tests(clss, basename+'_halflifted', fn, torch_fn, inputs, lift_every_other=True)
        if test_backward:
            add_backward_tests(clss, basename+'_halflifted', fn, torch_fn, inputs, lift_every_other=True)

def inputs(input_vals):
    return [TestInput(f'input{i}', vals) for i,vals in enumerate(input_vals)]

def random_inputs(input_shapes, rand=np.random.normal):
    return [TestInput(\
                reduce(lambda a,b: a + 'x' + b, [str(shape) for shape in shapes]),\
                [rand(size=shape) for shape in shapes]\
            )
        for shapes in input_shapes]