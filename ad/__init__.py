"""
The ad module contains all the code that implements the automatic differentiation
    engine; that is, the code that builds computation graphs and performs
    backpropagation on them
"""

import numpy as np
from collections.abc import Iterable 
from .node import Node

def value(x):
    """
    Extract the value from an autodiff Node.

    If you need to write code that operates on tensors, and the inputs to that code
        may be either numpy.ndarrays or Nodes, then calling this on the inputs will
        make sure they are all numpy.nadarrays.

    Args:
        x (any): any input value
    
    Returns:
        x.val if x is a Node, otherwise x
    """
    if isinstance(x, Node):
        return x.val
    else:
        return x