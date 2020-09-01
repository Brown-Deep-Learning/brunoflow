"""
This module contains code for defining new neural networks and their trainable parameters
"""

from ..ad import Node
from collections.abc import Iterable

class Parameter(Node):
    """
    A trainable parameter of a neural network layer.

    This is just syntactic sugar for a Node with no inputs.
    """

    def __init__(self, value):
        super(Parameter, self).__init__(value)

class Network:
    """
    A neural network with trainable parameters

    Attributes:
        parameters: a list of the trainable parameters for this layer

    Tensorflow analogue: tf.keras.layers.Layer / tf.keras.Model
        * Our Network class encompases concepts from both these keras classes.

    PyTorch analogue: torch.nn.Module
    """

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Perform the forward pass of the network layer
        """
        raise NotImplementedError('Layer __call__ method not implemented!')

    @property
    def parameters(self):
        """
        Retrieve the list of trainable parameters for this network
        """
        params = []
        # Get parameters from all attributes of this Network
        for key,val in vars(self).items():
            params.extend(_get_parameters(val))
        return params

def _get_parameters(val):
    if isinstance(val, Parameter):
        return [val]
    elif isinstance(val, dict):
        return [__get_parameters(v) for k,v in val.items()]
    elif isinstance(val, Iterable):
        return [__get_parameters(v) for v in val]
    # Recursively get parameters from Networks
    # (Allows compositionally building complex networks out of simple ones)
    elif isinstance(val, Network):
        return val.parameters
    else:
        return []