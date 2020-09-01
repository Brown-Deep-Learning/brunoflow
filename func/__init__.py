"""
The func module contains code for defining new autodiff functions (in function.py).

Its other submodules define autodiff versions of various common functions.

Tensorflow anaolgue:
    Most of the tf.* functions are differentiable by definition; this module defines
        analogues for many of these functions

PyTorch analogue:
    Most of the torch.* functions are differentiable by definition; this module defines
        analogues for many of these functions
    In addition, some of the functions defined in this module are analagous to those
        defined in torch.nn.functional
"""

from .function import *
from .activations import *
from .linalg import *
from .math import *
from .operators import *
from .reductions import *
from .shape import *