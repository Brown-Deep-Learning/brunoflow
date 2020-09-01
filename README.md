# brunoflow
A simple, well-documented, pedagogical deep learning framework implemented entirely in Python

## Design Philosophy
Brunoflow is a *pedagogical* programming framework: it's designed for those studying deep learning, not those who are putting it into practice in industry.
This target audience leads to several key design choices:

#### Transparency
Brunoflow aims to provide *transparent* implementations of the functionality commonly available in industry-standard deep learning frameworks such as Tensorflow and PyTorch.
Thus, Brunoflow is implemented *entirely in Python*: no calls to external C libraries that obfuscate what the functions are doing.
This makes it possible to inspect the implementations of forward and backward passes for every supported operation.
You can even easily fiddle with the implementation of the core automaic differentiation engine, if you like.

#### Interpretability Over Efficiency
Brunoflow prioritizes interpretability over efficiency.
It makes heavy use of numpy, including standard array-based programming idioms, but it makes no other attempt to be especially fast.
When multiple implementations of an algorithm exist, Brunoflow opts for the most readable one, rather than the most efficient one.
Also along these lines, Brunoflow has no GPU backend.

#### Transferability
Learners who familiarize themselves with Brunoflow should be able to transfer this knowledge to other, more "production-ready" deep learning frameworks.
Thus, wherever possible, the documentation for each Brunoflow function and class points out the equivalent (or analagous) entity in both Tensorflow and PyTorch.
Doing this consistently, throughout the entire codebase, helps drive home the universality of this concepts (and helps learners separate those concepts from the specifics of their implementation in any given framework).


## Package Structure
Deep learning is based on just three key ideas (plus a heap of little tricks to make things work well in practice):
1. Differentiable functions can be composed to produce more complex differentiable functions
1. A neural network is just a (composite) differentiable function, some of whose inputs are treated as optimizable parameters
1. Evaluating a composite differentiable function produces a *computation graph* which can be traversed backward to produce derivatives for gradient-based optimization.

Brunoflow's codebase is organized to make these ideas directly evident:
* **ad/**: This submodule implements construction of and backwards traversal of computation graphs to compute gradients.
* **func/**: This submodule implements many commonly-used differentiable functions and provides the logic for composing them.
* **net/**: This submodule implements neural networks by binding together differentiable functions and optimizable parameters into callable objects.
* **opt/**: This submodule implements common optimization objectives and common methods for gradient-based optimization of them.


## Credits
Brunoflow was designed and implemented for Brown CS 1470/2470 Deep Learning by Professor Daniel Ritchie with contributions from former Head TA Zachary Horvitz.
Brunoflow is free to use for all educational, research, and other non-commercial purposes (please consult the LICENSE).
If you use Brunoflow in your courses and/or research, please cite it as:
  
    @misc{Brunoflow,
        title = {{Brunoflow: A Pedagogical Deep Learning Framework}},
        author = {{Daniel Ritchie}},
        howpublished = {\url{https://github.com/Brown-Deep-Learning/brunoflow}},
        year = {2020}
    }

## The Name
Brunoflow is named after Brown University's mascot, Bruno the bear.
