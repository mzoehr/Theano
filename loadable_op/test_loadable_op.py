"""
    This is simple demonstration of the Theano loadable operator
"""

import sys
import numpy

import theano
import theano.tensor as T

from loadable import Loadable
#-----------------------------------------------------


class Data(object):
    """
      We define our data object
      This object should provide accessors to fetch a
      specific minibatch i from file or from memory.
      In this case we simply use the RAM
    """
    def __init__(self):
        self.N = 10
        self.inputs = [numpy.float32(numpy.arange(self.N) + i) for i in range(self.N)]
        self.targets = [numpy.float32(numpy.arange(self.N) + i) for i in range(self.N)]
        self.shared_inputs = theano.shared(self.inputs[0], name='inputs')
        self.shared_targets = theano.shared(self.targets[0], name='targets')

    def get_input(self, idx):
        # simulate file loading
        return self.inputs[idx]

    def get_target(self, idx):
        # simulate file loading
        return self.targets[idx]

#-----------------------------------------------------
# start the demo
#-----------------------------------------------------

# we create a Data object
data = Data()

# we create a Theano Loadable object (shared_memory + callback)
inputs = Loadable(data.shared_inputs, data.get_input, 'loadable_input')
targets = Loadable(data.shared_targets, data.get_target, 'loadable_target')

# we define a givens term
index = T.scalar()
x = T.vector('x')
y = T.vector('y')
givens = {x: inputs(index), y: targets(index)}

# we define a theano function
result = T.vector()
result = x + y
test_func = theano.function(inputs=[index], outputs=result, givens=givens)

# we run the test
for i in range(data.N):
    value = test_func(i)
    if  numpy.sum((value - (data.inputs[i] + data.targets[i]))) != 0:
        print 'computation incorrect', i
        print 'value', value
        print 'sum', data.inputs[i] + data.targets[i]
        print 'inputs', data.inputs[i]
        print 'targets', data.targets[i]
        sys.exit()

print 'computation correct'
