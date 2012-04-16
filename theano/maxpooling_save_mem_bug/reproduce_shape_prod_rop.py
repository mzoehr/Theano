import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample

#------------------------------------------------------------------------------
class RNN(object):
    def __init__(self, rng, input, output_taps, n_in, n_hidden, \
                 frames, name='RNN_', \
                 mode=theano.Mode(linker='cvm'), \
                 profile=0, dtype=theano.config.floatX):
        """
        """

        self.input = input
        # store rng
        self.rng = rng
        # length of output taps
        self.len_output_taps = len(output_taps)
        # create input weights and bias
        weights_in, bias_in = self.init_wb((n_in, n_hidden), dtype)
        # input to hidden layer weights
        W_in = theano.shared(weights_in, name=str(name) + 'W_in')
        # input bias
        b_in = theano.shared(bias_in, name=str(name) + 'b_in')
        # create recurrent weights and bias
        weights_r, _ = self.init_wb((n_hidden, n_hidden), dtype)
        # recurrent weights as real values
        W = [theano.shared(weights_r, name=str(name) + 'W_r' + str(u)) \
                            for u in range(self.len_output_taps)]

        # stack the network parameters
        self.params = []
        self.params.extend(W)
        self.params.extend([W_in, b_in])

        # recurrent activations
        h = theano.shared(numpy.zeros((frames, n_hidden)).astype(dtype), \
                               name=str(name) + '_r_act')

        # the hidden state `h` for the entire sequence, and the output for the
        # entry sequence `y` (first dimension is always time)
        y, updates = theano.scan(self.step,
                        sequences=input,
                        outputs_info=dict(initial=h, taps=output_taps),
                        non_sequences=self.params,
                        mode=mode,
                        profile=profile)

        # output of the network
        self.output = y

    #--------------------------------------------------------------------------
    def init_wb(self, size, dtype=theano.config.floatX):
        weights = 0.1 * numpy.asarray(self.rng.uniform(low=(-1.0), high=1.0, size=(size)))
        bias = numpy.zeros((size[1],))
        return (weights.astype(dtype), bias.astype(dtype))

    #--------------------------------------------------------------------------
    def step(self, u_t, *args):
            # get the recurrent activations
            r_act_vals = [args[u] for u in xrange(self.len_output_taps)]

            # get the recurrent weights
            r_weights = [args[u] for u in range(self.len_output_taps, \
                                                (self.len_output_taps) * 2)]
            # get the input/output weights
            W_in = args[self.len_output_taps * 2]
            b_in = args[self.len_output_taps * 2 + 1]
            # sum up the recurrent activations
            act = T.dot(r_act_vals[0], r_weights[0])
            # compute the new recurrent activation
            h_t = T.tanh(T.dot(u_t, W_in) + b_in + act)

            return h_t

#------------------------------------------------------------------------------
class OutputLayer(object):

    #--------------------------------------------------------------------------
    def __init__(self, rng, input, n_in, n_out,
                 mode=theano.Mode(linker='cvm'),
                 profile=0, dtype=theano.config.floatX):

        #-----------------------------------------
        # parameter STUFF
        #-----------------------------------------
        self.rng = rng

        #-----------------------------------------
        # SOFTMAX LAYER STUFF
        #-----------------------------------------
        # create weights and bias
        weights, bias = self.init_wb((n_in, n_out))
        # set share weights
        W = theano.shared(value=weights, name='W_layer')
        # set shared bias
        b = theano.shared(value=bias, name='b_layer')

        # we have to support minibatches
        if(input.ndim == 3):

            # theano has no tensor support at the moment
            y, updates = theano.scan(self.softmax_tensor,
                    sequences=input,
                    non_sequences=[W, b],
                    mode=mode,
                    profile=profile)

            # compute vector of class-membership probabilities in symbolic form
            self.p_y_given_x = y.reshape((input.shape[0] * input.shape[1], n_out))
            # output activations (different due to tensor3
            self.output = y

        else:

            # compute vector of class-membership probabilities in symbolic form
            self.p_y_given_x = T.nnet.softmax(T.dot(input, W) + b)
            # output activations
            self.output = self.p_y_given_x

        # compute prediction as class whose probability is maximal in symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [W, b]

    #--------------------------------------------------------------------------
    def softmax_tensor(self, h, W, b):
        return T.nnet.softmax(T.dot(h, W) + b)

    #--------------------------------------------------------------------------
    def negative_log_likelihood(self, y):
        return -T.sum(T.log(self.p_y_given_x + 1e-14)[T.arange(y.shape[0]), y])

    #--------------------------------------------------------------------------
    def init_wb(self, size, dtype=theano.config.floatX):
        weights = 0.1 * numpy.asarray(self.rng.uniform(low=(-1.0), high=1.0, size=(size)))
        bias = numpy.zeros((size[1],))
        return (weights.astype(dtype), bias.astype(dtype))

#--------------------------------------------------------------------------
class Engine(object):

    def __init__(self,
                learning_rate=0.01,
                n_epochs=1,
                output_taps=[-1]):

        #-----------------------------------------
        # THEANO SETUP
        #-----------------------------------------
        # setup mode
        mode = 'DEBUG_MODE'
        #mode = theano.Mode(linker='cvm')
        # setup profile
        profile = 0
        # theano dtype
        dtype = theano.config.floatX

        #-----------------------------------------
        # MODEL PARAMETERS
        #-----------------------------------------
        # number of samples
        N = 10
        # number of input units
        n_in = 784
        # number of hidden units
        n_hidden = [100, 100]
        # number of output units
        n_out = 10
        # sequence length
        length = 10
        # downsample
        maxpooling = [1, 2]
        # initialize random generator
        rng = numpy.random.RandomState(123467)

        #-----------------------------------------
        # DATA SETUP (simply random data)
        #-----------------------------------------
        data_x = numpy.random.randn(N, length, n_in).astype('float32')
        #data_y = numpy.random.randint(0, n_out, (N * length)).astype('int8')
        print '... input shape: {} '.format(data_x.shape)
        #print '... output shape: {} '.format(data_y.shape)

        #-----------------------------------------
        # SETUP MODEL
        #-----------------------------------------
        #-----------------------------------------
        # Theano variables
        #-----------------------------------------
        # input (where first dimension is time)
        self.u = T.tensor3()
        # target (where first dimension is time)
        self.t = T.bvector()
        # learning rate
        self.lr = T.scalar()

        #-----------------------------------------
        # create the RNN layers
        #-----------------------------------------
        self.rnnLayers = []
        for layer in range(len(n_hidden)):

            # number of input neurons to the layer
            if(layer == 0):
                inputs = self.u
                num_inputs = n_in
            else:
                inputs = self.rnnLayers[-1].output
                num_inputs = n_hidden[layer - 1]

                # enable downsampling
                if(maxpooling != None):
                    print '... downsampling enabled'
                    inputs = downsample.max_pool_2d(inputs, maxpooling, ignore_border=False)
                    num_inputs = num_inputs / maxpooling[1]

            # create a RNN layer (forward direction)
            rnnLayer = RNN(rng=rng, input=inputs, output_taps=output_taps, n_in=num_inputs, \
                                n_hidden=n_hidden[layer], \
                                frames=data_x.shape[1], \
                                mode=mode, profile=profile, name='RNN_' + str(layer))
            # push the layers
            self.rnnLayers.append(rnnLayer)

        #-----------------------------------------
        # construct the softmax output layer
        #-----------------------------------------
        self.outputLayer = OutputLayer(rng=rng, input=self.rnnLayers[-1].output,
                                 n_in=n_hidden[-1], n_out=n_out)

        # define the cost to minimize
        self.cost = self.outputLayer.negative_log_likelihood(self.t)

        # define the output of the model
        output = self.rnnLayers[-1].output #self.outputLayer.output

        # add the network parameters 
        params = []
        params.extend(self.outputLayer.params)
        [params.extend(layer.params)  for layer in self.rnnLayers]

        print '... network: n_in:{}, n_hidden:{}, n_out:{}, output:softmax, maxpooling:{}'\
                .format(n_in, n_hidden, n_out, maxpooling)

        #-----------------------------------------
        # DEMO FUNCTION
        #-----------------------------------------
        givens = {self.u: data_x}
        grhs = [theano.shared(numpy.zeros(p.get_value().shape, dtype=dtype), borrow=True) for p in params]
        func = theano.function(inputs=[], outputs=T.Rop(output, params, grhs), givens=givens, mode=mode, profile=profile)

        #-----------------------------------------
        # DEMO START
        #-----------------------------------------
        func()
        # BOOM!!!

#------------------------------------------------------------------------------
if __name__ == '__main__':

    Engine()

