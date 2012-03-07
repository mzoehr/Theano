import theano

"""
    The Theano loadable operator is designed to remove the
    dependency between a python Data object, storing inputs and targets,
    and theano function calls.

    Since graphic cards have limited memory, very often it
    is not possible to store the complete dataset (all minibatches)
    in a single theano shared memory. A data object is needed to
    load the specified minibatch into the shared_mem, when accessed.

    When using more advanced optimizer classes you have to pass this data
    object as a parameter (CPU) making this class dependent on the underlying
    object. The code is not very portable and reusable anymore! Manually
    calling an external function to update a shared variable between each call
    makes the code more cumbersome as well.

    However the use of the Loadable operator will help you to overcome this
    problem. Since more advanced python optimizer classes need at least an
    input an a target, which is passed with givens as a function parameter
    and a model output (theano) you can define a Loadable like:

        # we create a Theano Loadable object (shared_memory + callback)
        inputs = Loadable(data.get_input, 'l_input')
        targets = Loadable(data.get_target, 'l_target')

        # we define a givens term
        index = T.scalar()
        x = T.vector('x')
        y = T.vector('y')
        givens = {x: inputs(index), y: targets(index)}

    where data.get_XXX is a function of your data object returning a minibatch
    from CPU mem or file. Whenever a theano function will access givens the
    OP will return the updated shared_mem for the given minibatch.

    Hence you are able to pass givens and your model output to a optimizer
    class, but NOT the data object, making your code a lot more portable and
    transparent =)

        optimizer = Optimizer(givens, output)

"""


class Loadable(theano.Op):

    def __init__(self, fn, name='LoadableOp'):
        self.name = name
        self.fn = fn
        self.index = 0
        self.input = self.fn(self.index)

    def make_node(self, index):
        index = theano.tensor.as_tensor_variable(index)
        if(self.input.ndim == 1):
            return theano.Apply(self,
                            inputs=[index],
                            outputs=[theano.tensor.vector()])
        elif(self.input.ndim == 2):
            return theano.Apply(self,
                            inputs=[index],
                            outputs=[theano.tensor.matrix()])
        elif(self.input.ndim == 3):
            return theano.Apply(self,
                            inputs=[index],
                            outputs=[theano.tensor.tensor3()])
        elif(self.input.ndim == 4):
            return theano.Apply(self,
                            inputs=[index],
                            outputs=[theano.tensor.tensor4()])
        else:
            raise TypeError('%s only works on vector/matrix/tensor3/tensor4' \
                            % self.name)

    def __eq__(self, other):
        return type(self) == type(other) and \
               (self.name == other.name) and \
               (self.fn == other.fn)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.name) ^ \
               hash(self.fn)

    def __str__(self):
        return self.name

    def perform(self, node, inputs_storage, output_storage):
        if(self.index != inputs_storage[0]):
            self.index = int(inputs_storage[0])
            self.input = self.fn(self.index)
        output_storage[0][0] = self.input.copy()
