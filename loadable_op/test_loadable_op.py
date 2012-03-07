"""
    This is simple demonstration of the Theano loadable operator
"""

import sys
import scipy

import theano
import theano.tensor as T
from theano.tensor import TensorType

from loadable_array import Loadable
#-----------------------------------------------------


class Data(object):
    """
      We define our data object
      This object should provide accessors to fetch a
      specific minibatch i from file or from memory.
      In this case we simply use the RAM

      The shape of the data could be list of
      vectors or matrix.

      typical minibatch definition: [N,seq_len,frame]
    """
    def __init__(self, shape=(5, 5), N=10):

        self.N = N
        self.inputs = [scipy.random.random(shape).astype(theano.config.floatX) for _ in range(self.N)]
        self.targets = [scipy.random.random(shape).astype(theano.config.floatX) for _ in range(self.N)]

    def get_input(self, idx):
        # simulate file loading
        return self.inputs[idx]

    def get_target(self, idx):
        # simulate file loading
        return self.targets[idx]

#-----------------------------------------------------
# define the parameters
#-----------------------------------------------------
N = 100
shapes = [5, (5, 5), (5, 5, 5), (5, 5, 5, 5)]
types = [T.vector, T.matrix, T.tensor3, T.tensor4]

#-----------------------------------------------------
# start the demo
#-----------------------------------------------------
for shape, type in zip(shapes, types):

    # we create a Data object
    data = Data(shape, N)

    # we create a Theano Loadable object (shared_memory + callback)
    inputs = Loadable(data.get_input, 'loadable_input')
    targets = Loadable(data.get_target, 'loadable_target')

    # we define a givens term
    index = T.scalar()
    x = type('x')
    y = type('y')
    givens = {x: inputs(index), y: targets(index)}

    # we define a theano function
    result = T.vector()
    result = x + y
    test_func = theano.function(inputs=[index], outputs=result, givens=givens)

    # we run the test
    for i in range(data.N):
        value = test_func(i)
        if  scipy.sum((value - (data.inputs[i] + data.targets[i]))) != 0:
            print 'computation incorrect', i
            print 'value', value
            print 'sum', data.inputs[i] + data.targets[i]
            print 'inputs', data.inputs[i]
            print 'targets', data.targets[i]
            sys.exit()

    print 'computation correct for shape', shape
