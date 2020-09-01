import numpy as np

class Node:
    """
    Represents one node in an autodiff computation graph.

    Each Node stores a value and a gradient.
    Nodes are created by calling autodiff functions (the forward pass)
    The gradient of each Node is computed during the backward passs.

    Tensorflow analogue:
        tf.Tensor

    PyTorch analogue:
        torch.Tensor

    Attributes:
        val (numpy.ndarray | float): The value at this point in the forward pass over the
            computation graph
        grad (numpy.ndarray | float): The gradient at this point in the backward pass over
            the computation graph
        inputs (list): The inputs to the function which produced this Node as output
            (empty for leaf Nodes). These may be Nodes themselves, np.ndarrays, or other types
        backward_func: A function which takes self.grad and computes gradients for each
            of self.inputs (None for leaf Nodes)
        shape (tuple): The same as numpy.ndarray.shape
        size (int): The same as numpy.ndarray.size
        ndim (int): The same as numpy.ndarray.ndim
    """

    def __init__(self, val, backward_func=None, inputs=[]):
        self.val = val
        self.grad = np.zeros_like(val,dtype=np.float64)
        self.backward_func = backward_func
        self.inputs = inputs
        self.__num_uses = 0

    @property
    def shape(self):
        return self.val.shape
    
    @property
    def size(self):
        return self.val.size

    @property
    def ndim(self):
        return self.val.ndim

    def __str__(self):
        return f'node(val: {self.val}, grad: {self.grad})'
    def __repr__(self):
        return str(self)

    def __compute_num_uses(self):
        """
        Starting at this node, recursively compute how many times each Node is used as an
            input by some other Node in the graph.
        This is used during backpropagation to determine when all the gradients for the
            Node have been accmulated and the backward pass can move on to its inputs
        """
        self.__num_uses += 1
        if self.__num_uses == 1:
            for inp in self.inputs:
                if isinstance(inp, Node):
                    inp.__compute_num_uses()

    def backprop(self):
        """
        Initiate a backward pass starting from this Node.
        Computes gradients from every Node reachable from this Node, back to the leaves.
        """
        if isinstance(self.grad, np.ndarray):
            self.grad.fill(1.0)
        else:
            # If the grad is a float rather than an np.ndarray
            self.grad = 1.0
        self.__compute_num_uses()
        self.__backprop()

    def __backprop(self):
        """
        Recursive helper function for self.backprop()
        Assumes that self.__compute_num_uses() has been called in advance
        """
        # Record that backprop has reached this Node one more time
        self.__num_uses -= 1
        # If backprop has reached this Node as many times as it has been used as an input,
        #   then all of its gradients have been accmulated and backprop can continue up the graph.
        if self.__num_uses == 0:
            backward_func = self.backward_func
            if backward_func:
                input_vals = [inp.val if isinstance(inp, Node) else inp for inp in self.inputs]
                # The 'adjoint' is the partial derivative of the final node in the graph
                #   (typically the loss) w.r.t. the value at this Node.
                adjoints = backward_func(self.val, self.grad, *input_vals)
                assert len(input_vals) == len(adjoints)
                # Accumulate these adjoints into the gradients for this Node's inputs
                for i,adj in enumerate(adjoints):
                    if isinstance(self.inputs[i], Node):
                        # An adjoint may need to be 'reshaped' before it can be accumulated
                        #   if the forward operation used broadcasting.
                        self.inputs[i].grad += reshape_adjoint(adj, self.inputs[i].grad.shape)
            # Continue recursively backpropagating
            for inp in self.inputs:
                if isinstance(inp, Node):
                    inp.__backprop()

    def zero_gradients(self):
        """
        Zero out the gradients for all Nodes reachable from this Node (including this Node).
        Typically called before performing an optimization step, to ensure that gradients
            do not spuriously accumulate across iterations of optimization.
        """
        self.__compute_num_uses()
        self.__zero_gradients()

    def __zero_gradients(self):
        """
        Recursive helper for self.zero_gradients
        Assumes that self.__compute_num_uses() has been called in advance
        """
        # We use self.__num_uses to ensure that we traverse each Node in the graph once.
        # Since all we're doing is zeroing out gradients, it technically wouldn't be
        #    incorrect to traverse them multiple times, but it could be a lot less efficient.
        self.__num_uses -= 1
        if self.__num_uses == 0:
            if isinstance(self.grad, np.ndarray):
                self.grad.fill(0.0)
            else:
                self.grad = 0.0
            for inp in self.inputs:
                if isinstance(inp, Node):
                    inp.__zero_gradients()


def reshape_adjoint(adj, shape):
    """
    Convert an adjoint into the appropriate shape before accumulating it into a Node's gradient.

    This is needed whenever a forward operation in the graph performs broadcasting.
    Consider adding a tensor of shape (4, 3, 3) to a tensor of shape (3, 3):

    (4, 3, 3) + (3, 3) --> (4, 3, 3) + (1, 3, 3) --> (4, 3, 3)  [via broadcasting]

    The gradient at the output node will have shape (4, 3, 3).
    Computing a point-wise adjoint for the input gives a gradient of shape (4, 3, 3), but the input
        gradient expects a tensor of size (3, 3).
    Thus, we need to sum-reduce the adjoint along the 0-th axis to produce a (3, 3) adjoint.

    Why is sum-reduction the correct thing to do? Broadcasting could be explicitly represented in 
        the computation graph as a 'stack' operation that takes in 4 copies of the (3, 3) input
        tensor and produces the (4, 3, 3) output tensor by stacking them along the 0-th axis.
    The stack operation uses the (3, 3) input 4 times, which would result in the adjoint it produces
        being accumulated into the gradient of that input 4 times, which is equivalent to the sum-
        reduction performed by this function.
    In otherwords: this function is a shortcut for adding a 'stack' operation into the graph
        whenever broadcasting occurs.

    Args:
        adj: the adjoint tensor to be reshaped.
        shape: the new shape that the adjoint tensor should have after 'un-broadcasting'

    Returns:
        adj, summed along dimensions which were produced via broadcasting.
    """
    if adj.shape == shape:
        return adj
    # Should never happen (because broadcasting only adds dimensions; it doesn't remove them)
    if len(shape) > len(adj.shape):
        raise ValueError(f'Cannot reshape adjoint with shape {adj.shape} to accumulate into shape {shape}')
    # Sum along matching dimensions
    for i in range(1, len(shape)+1):
        if adj.shape[-i] != shape[-i]:
            if shape[-i] == 1:
                adj = np.sum(adj, axis=-i, keepdims=True)
            else:
                raise ValueError(f'Cannot reshape adjoint with shape {adj.shape} to accumulate into shape {shape}')
    # Sum along extra dimensions that adj has
    if len(adj.shape) > len(shape):
        for i in range(len(adj.shape) - len(shape)):
            adj = np.sum(adj, axis=0)
    return adj
