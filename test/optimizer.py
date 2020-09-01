import unittest as ut
import brunoflow as bf
import numpy as np

m = 3.4
b = 1.2

def linreg(opt_ctor):
    # Training data
    n = 50
    x = np.random.normal(size=[n])
    y = m*x + b

    # Model
    m_p = bf.Parameter(np.random.normal())
    b_p = bf.Parameter(np.random.normal())
    def f(x):
        return m_p*x + b_p

    # Training loop
    optimizer = opt_ctor([m_p, b_p])
    iters = 3000
    for i in range(iters):
        loss = bf.opt.mse_loss(f(x), y)
        optimizer.zero_gradients()
        loss.backprop()
        optimizer.step()

    return m_p.val, b_p.val

class OptimizerTestCase(ut.TestCase):

    def check(self, val, target):
        if not np.allclose(val, target, rtol=0.05, atol=0):
            self.assertIs(val, target)
        
    def test_sgd(self):
        m_p, b_p = linreg(lambda params: bf.opt.SGD(params, step_size=0.1))
        self.check(m_p, m)
        self.check(b_p, b)

    def test_adagrad(self):
        m_p, b_p = linreg(lambda params: bf.opt.AdaGrad(params, step_size=0.1))
        self.check(m_p, m)
        self.check(b_p, b)

    def test_rmsprop(self):
        m_p, b_p = linreg(lambda params: bf.opt.RMSProp(params, step_size=0.1))
        self.check(m_p, m)
        self.check(b_p, b)

    def test_adam(self):
        m_p, b_p = linreg(lambda params: bf.opt.Adam(params, step_size=0.1))
        self.check(m_p, m)
        self.check(b_p, b)
