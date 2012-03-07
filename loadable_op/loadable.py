import theano

"""
    The Theano loadable operator is designed to remove the
    dependency between a python Data object, storing inputs and targets,
    and theano functions.

    Since graphic cards have limited memory, very often it
    is not possible to store the complete dataset (all minibatches)
    in a single theano shared memory. A data object is needed to 
    load the specified minibatch into the shared_mem, when accessed.

    When using more advanced optimizer classes you have to pass this data object
    as a parameter making this class dependent on the data object. The code is not very
    portable and reusable anymore! Manually calling an external
    function to update a shared variable between each call makes the code more
    cumbersome as well.

    However using the Loadable operator will help you to overcome this problem.
    Since more advanced python optimizer classes need at least a input an a target,
    which is passed with givens as a function parameter and a model output (theano)
    you can define a Loadable like:

        # we create a Theano Loadable object (shared_memory + callback)
        inputs = Loadable(data.shared_inputs, data.get_input, 'loadable_input')
        targets = Loadable(data.shared_targets, data.get_target, 'loadable_target')

        # we define a givens term
        index = T.scalar()
        x = T.vector('x')
        y = T.vector('y')
        givens = {x: inputs(index), y: targets(index)}

    where data.get_XXX is a function of your data object loading a minibatch from CPU mem
    or file, and data.shared_XXX is your shared memory object (storing 1 minibatch)
    Whenever a theano function will access givens the OP will return the updated
    shared_mem for the given minibatch.

    Hence you are able to pass givens and your model output to a optimizer class, 
    but NOT the data object, making your code a lot more portable =)

        optimizer = Optimizer(givens, output)

"""


class Loadable(theano.Op):

    def __init__(self, shared, fn, name=None):
        self.name = name
        self.fn = fn
        self.shared = shared
        self.index = 0

    def make_node(self, index):
        index = theano.tensor.as_tensor_variable(index)
        return theano.Apply(self,
                            inputs=[index],
                            outputs=[theano.tensor.vector()])

    def __eq__(self, other):
        return type(self) == type(other) and \
               (self.shared == other.shared) and \
               (self.name == other.name) and \
               (self.fn == other.fn)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.name) ^ \
               hash(self.shared) ^ hash(self.fn)

    def __str__(self):
        return self.name

    def perform(self, node, inputs_storage, output_storage):
        if(self.index != inputs_storage[0]):
            self.index = int(inputs_storage[0])
            self.shared.set_value(self.fn(self.index))
        output_storage[0][0] = self.shared.get_value()
