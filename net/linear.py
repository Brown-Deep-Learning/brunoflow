import math
import numpy as np
from .network import Network, Parameter
from ..func import matmul

def xavier2(m, n):
    return math.sqrt(2 / (m + n))

def xavier1(n):
    return math.sqrt(1/n)

class Linear(Network):

    def __init__(self, input_size, output_size):
        m = input_size
        n = output_size
        self.W = Parameter(np.random.normal(scale=xavier2(m, n), size=(m, n)))
        self.b = Parameter(np.random.normal(scale=xavier1(n), size=(n,)))

    def forward(self, x):
        return matmul(x, self.W) + self.b