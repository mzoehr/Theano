"""
    This is simple demonstration of the Theano loadable operator
"""

from nose.plugins.skip import SkipTest
import sys

import numpy as np

import theano
import theano.tensor as T
from theano.tensor import TensorType
from theano.misc.may_share_memory import may_share_memory
from loadable import Loadable
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
        self.inputs = [np.random.random(shape).astype(theano.config.floatX)
                       for _ in range(self.N)]
        self.targets = [np.random.random(shape).astype(theano.config.floatX)
                        for _ in range(self.N)]

    def get_input(self, idx):
        # simulate file loading
        return self.inputs[idx]

    def get_target(self, idx):
        # simulate file loading
        return self.targets[idx]


def test_loadable():
    #-----------------------------------------------------
    # define the parameters
    #-----------------------------------------------------
    N = 100
    shapes = [5, (5, 6), (4, 5, 6), (3, 4, 5, 6)]
    types = [T.vector, T.matrix, T.tensor3, T.tensor4]

    #-----------------------------------------------------
    # start the demo
    #-----------------------------------------------------
    for shape, typ in zip(shapes, types):

        # we create a Data object
        data = Data(shape, N)

        # we create a Theano Loadable object (shared_memory + callback)
        inputs = Loadable(data.get_input, 'input')
        targets = Loadable(data.get_target, 'target')

        # we define a givens term
        index = T.scalar()
        x = typ('x')
        y = typ('y')
        givens = {x: inputs(index), y: targets(index)}

        # we define a theano function
        result = T.vector()
        result = x + y
        test_func = theano.function(inputs=[index], outputs=[result, x, y],
                                    givens=givens)
        theano.printing.debugprint(test_func)

        # we run the test
        for i in range(data.N):
            values = test_func(i)
            if  np.sum((values[0] - (data.inputs[i] + data.targets[i]))) != 0:
                print 'computation incorrect', i
                print 'value', value
                print 'sum', data.inputs[i] + data.targets[i]
                print 'inputs', data.inputs[i]
                print 'targets', data.targets[i]
                assert False
            values2 = test_func(i)
            for i in range(3):
                assert not may_share_memory(values2[i], values[i])
                assert (np.asarray(values2[i]) == np.asarray(values[i])).all()
        print 'computation correct for shape', shape
