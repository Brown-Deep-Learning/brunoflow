import unittest as ut
import brunoflow as bf
import numpy as np

class NetworkTestCase(ut.TestCase):

    def check(self, val, target):
        err_thresh = 0.05
        diff_norm = np.mean(np.abs(val - target))
        rel_diff_norm = diff_norm / np.mean(np.abs(target))
        self.assertLess(rel_diff_norm, err_thresh)
        
    def test_linear(self):
        in_dim = 10
        out_dim = 4

        m = np.full(shape=(in_dim, out_dim), fill_value=0.2)
        b = np.full(shape=(out_dim), fill_value=-0.1)

        # Training data
        n = 50
        x = np.random.normal(size=[n, in_dim])
        y = x @ m + b

        # Model
        linear = bf.net.Linear(in_dim, out_dim)

        # Training loop
        optimizer = bf.opt.SGD(linear.parameters, step_size=0.1)
        iters = 1000
        for i in range(iters):
            loss = bf.opt.mse_loss(linear(x), y)
            optimizer.zero_gradients()
            loss.backprop()
            optimizer.step()

        self.check(linear.W.val, m)
        self.check(linear.b.val, b)