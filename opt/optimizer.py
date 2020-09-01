"""
This module implements common optimization methods
"""

import numpy as np

class Optimizer:
    """
    Represents an optimization method.

    Attributes:
        params: a list of Parameters to be optimized
        step_size: gradient step size hyperparameter
    """

    def __init__(self, params_to_optimize, step_size=0.001):
        self.params = params_to_optimize
        self.step_size = step_size

    def step(self):
        """
        Take one optimization step
        """
        raise NotImplementedError('Optimizer step method not implemented!')

    def zero_gradients(self):
        """
        Zero out the gradients for all parameters to be optimzed.
        Must be called before computing the backward pass to prevent incorrect gradient
            accumulation across ierations.
        """
        for p in self.params:
            p.zero_gradients()


# --------------------------------------------------------------------------------


class SGD(Optimizer):

    def __init__(self, params_to_optimize, step_size=0.001, momemtum=0.0):
        super(SGD, self).__init__(params_to_optimize, step_size=step_size)
        self.mu = momemtum
        self.v = [np.zeros_like(p.val, dtype=float) for p in params_to_optimize]

    def step(self):
        for i,p in enumerate(self.params):
            self.v[i] = self.v[i] * self.mu - p.grad * self.step_size
            p.val += self.v[i]


# --------------------------------------------------------------------------------


class AdaGrad(Optimizer):

    def __init__(self, params_to_optimize, step_size=0.001, eps=1e-8):
        super(AdaGrad, self).__init__(params_to_optimize, step_size)
        self.eps = eps
        self.g2 = [np.zeros_like(p.val, dtype=float) for p in params_to_optimize]

    def step(self):
        for i,p in enumerate(self.params):
            self.g2[i] += p.grad*p.grad
            p.val -= self.step_size * (p.grad / (np.sqrt(self.g2[i]) + self.eps))


# --------------------------------------------------------------------------------


class RMSProp(Optimizer):

    def __init__(self, params_to_optimize, step_size=0.001, decay_rate=0.9, eps=1e-8):
        super(RMSProp, self).__init__(params_to_optimize, step_size)
        self.decay_rate = decay_rate
        self.eps = eps
        self.g2 = [np.zeros_like(p.val, dtype=float) for p in params_to_optimize]

    def step(self):
        for i,p in enumerate(self.params):
            self.g2[i] = self.decay_rate*self.g2[i] + (1-self.decay_rate)*p.grad*p.grad
            p.val -= self.step_size * (p.grad / (np.sqrt(self.g2[i]) + self.eps))


# --------------------------------------------------------------------------------


class Adam(Optimizer):

    def __init__(self, params_to_optimize, step_size=0.001, beta1=0.9, beta2=0.99, eps=1e-8):
        super(Adam, self).__init__(params_to_optimize, step_size)
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = [np.zeros_like(p, dtype=float) for p in params_to_optimize]
        self.v = [np.zeros_like(p, dtype=float) for p in params_to_optimize]

    def step(self):
        for i,p in enumerate(self.params):
            self.m[i] = self.beta1*self.m[i] + (1-self.beta1)*p.grad
            self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*p.grad*p.grad
            m = self.m[i] / (1 - self.beta1)
            v = self.v[i] / (1 - self.beta2)
            p.val -= self.step_size * (m / (np.sqrt(v) + self.eps))
