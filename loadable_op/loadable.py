import theano


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
