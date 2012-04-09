import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample

#------------------------------------------------------------------------------
class RNN(object):
    def __init__(self, rng, input, output_taps, n_in, n_hidden, \
                 frames, direction='forward',
                 W=None, b=None, params=None, name='RNN_', \
                 mode=theano.Mode(linker='cvm'), \
                 profile=0, dtype=theano.config.floatX):
        """
        """

        # store rng
        self.rng = rng

        # specify RNN direction
        if(direction == 'forward'):
            direction = False
        elif(direction == 'backward'):
            direction = True
        else:
            raise NotImplementedError("Invalid RNN direction specified \
                                       [forward|backward]")

        # length of output taps
        self.len_output_taps = len(output_taps)

        # load the saved network parameters if given
        if((W != None) and (b != None)):
            W_in = W
            b_in = b
            W = [theano.shared(self.make_W((n_hidden, n_hidden)), \
                               name=str(name) + 'W_r' + str(u)) for u in range(self.len_output_taps)]
        elif(params == None):
            # create input weights and bias
            weights_in, bias_in = self.init_wb((n_in, n_hidden), dtype)
            # input to hidden layer weights
            W_in = theano.shared(weights_in, name=str(name) + 'W_in')
            # input bias
            b_in = theano.shared(bias_in, name=str(name) + 'b_in')

            # create recurrent weights and bias
            weights_r, _ = self.init_wb((n_hidden, n_hidden), dtype)
            # recurrent weights as real values
            W = [theano.shared(weights_r, name=str(name) + 'W_r' + str(u)) for u in range(self.len_output_taps)]

        else:
            W_in = theano.shared(params[str(name) + 'W_in'], name=str(name) + 'W_in')
            b_in = theano.shared(params[str(name) + 'b_in'], name=str(name) + 'b_in')
            W = [theano.shared(params[str(name) + 'W_r' + str(u)], name=str(name) + 'W_r' + str(u)) for u in range(self.len_output_taps)]

        # stack the network parameters
        self.params = []
        self.params.extend(W)
        self.params.extend([W_in, b_in])

        # recurrent activations
        self.h = theano.shared(numpy.zeros((frames, n_hidden)).astype(dtype), name=str(name) + '_r_act')

        # the hidden state `h` for the entire sequence, and the output for the
        # entry sequence `y` (first dimension is always time)
        y, updates = theano.scan(self.step,
                        sequences=input,
                        outputs_info=dict(initial=self.h, taps=output_taps),
                        non_sequences=self.params,
                        go_backwards=direction,
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
            """
                step function to calculate BPTT

                type u_t: T.matrix()
                param u_t: input sequence of the network

                type * args: python parameter list

            """

            # get the recurrent activations
            r_act_vals = [args[u] for u in xrange(self.len_output_taps)]

            # get the recurrent weights
            r_weights = [args[u] for u in range(self.len_output_taps, (self.len_output_taps) * 2)]

            # get the input/output weights
            W_in = args[self.len_output_taps * 2]
            b_in = args[self.len_output_taps * 2 + 1]

            # sum up the recurrent activations
            act = T.dot(r_act_vals[0], r_weights[0])
            # no support @ the moment
            #for u in xrange(1, self.len_output_taps):
            #    act += T.dot(r_act_vals[u], r_weights[u])

            # compute the new recurrent activation
            h_t = T.tanh(T.dot(u_t, W_in) + b_in + act)

            return h_t

#------------------------------------------------------------------------------
class OutputLayer(object):

    #--------------------------------------------------------------------------
    def __init__(self, rng, input, n_in, n_out,
                 W=None, b=None,
                 mode=theano.Mode(linker='cvm'),
                 profile=0, dtype=theano.config.floatX,
                 params=None):

        #-----------------------------------------
        # parameter STUFF
        #-----------------------------------------
        self.rng = rng

        #-----------------------------------------
        # SOFTMAX LAYER STUFF
        #-----------------------------------------
        if((W != None) and (b != None)):
            self.W = W
            self.b = b
        elif(params == None):
            # create weights and bias
            weights, bias = self.init_wb((n_in, n_out))
            # set share weights
            self.W = theano.shared(value=weights, name='W_layer')
            # set shared bias
            self.b = theano.shared(value=bias, name='b_layer')
        else:
            self.W = theano.shared(params['W_layer'], name='W_layer')
            self.b = theano.shared(params['b_layer'], name='b_layer')

        # we have to support minibatches
        if(input.ndim == 3):

            # theano has no tensor support at the moment
            y, updates = theano.scan(self.softmax_tensor,
                    sequences=input,
                    non_sequences=[self.W, self.b],
                    mode=mode,
                    profile=profile)

            # compute vector of class-membership probabilities in symbolic form
            self.p_y_given_x = y.reshape((input.shape[0] * input.shape[1], n_out))
            # output activations (different due to tensor3
            self.output = y

        else:
            # compute vector of class-membership probabilities in symbolic form
            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
            # output activations
            self.output = self.p_y_given_x

        # compute prediction as class whose probability is maximal in symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    #--------------------------------------------------------------------------
    def softmax_tensor(self, h, W, b):
        return T.nnet.softmax(T.dot(h, W) + b)

    #--------------------------------------------------------------------------
    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        """
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
        mode = theano.Mode(linker='cvm') #'DEBUG_MODE' 
        # setup profile
        profile = 0

        #-----------------------------------------
        # MODEL PARAMETERS
        #-----------------------------------------
        # number of samples
        N = 1000
        # number of input units
        n_in = 784
        # number of hidden units
        n_hidden = [100, 100]
        # number of output units
        n_out = 10
        # sequence length
        length = 10
        # batch_size
        batch_size = 100
        # downsample
        maxpooling = [1, 2]
        # initialize random generator
        rng = numpy.random.RandomState(123467)

        #-----------------------------------------
        # DATA SETUP (simply random data)
        #-----------------------------------------
        data_x = numpy.random.randn(N, length, n_in).astype('float32')
        data_y = numpy.random.randint(0, n_out, (N * length)).astype('int8')
        print '... input shape: {} '.format(data_x.shape)
        print '... output shape: {} '.format(data_y.shape)

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
                                frames=data_x.shape[1], direction='forward', \
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

        # add the network parameters 
        self.params = []
        self.params.extend(self.outputLayer.params)
        [self.params.extend(layer.params)  for layer in self.rnnLayers]

        print '... network: n_in:{}, n_hidden:{}, n_out:{}, output:softmax, maxpooling:{}'\
                .format(n_in, n_hidden, n_out, maxpooling)

        #-----------------------------------------
        # THEANO train function
        #-----------------------------------------
        gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            gparams.append(gparam)

        # specify how to update the parameters of the model as a dictionary
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - self.lr * gparam

        # compiling a Theano function `train_fn` that returns the cost, but
        # in the same time updates the parameter of the model based on the
        # rules defined in `updates`
        train_fn = theano.function(inputs=[self.u, self.t],
                outputs=self.cost,
                updates=updates,
                givens={self.lr: T.cast(0.001, 'float32')},
                mode=mode,
                profile=profile,
                allow_input_downcast=True)

        #-----------------------------------------
        # DEMO START
        #-----------------------------------------
        # start the benchmark
        start_time = time.clock()
        print 'Running ({} epochs)'.format(n_epochs)
        for _ in xrange(n_epochs):
            for index in range(0, N / batch_size):
                train_fn(data_x[index * batch_size:(index + 1) * batch_size],
                         data_y[index * (batch_size * length):(index + 1) * (batch_size * length)])

        #print >> sys.stderr, ('     training epoch time (%.5fm)' % \
        #                      ((time.clock() - start_time) / 60.))

#------------------------------------------------------------------------------
if __name__ == '__main__':

    Engine()

