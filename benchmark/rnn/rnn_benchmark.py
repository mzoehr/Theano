import sys
import os
import time
import numpy
import theano
import theano.tensor as T

#---------------------------------------------------------------------------------
class RNN(object):

    #---------------------------------------------------------------------------------
    def __init__(self, rng, order, n_in, n_hidden, n_out, dtype=theano.config.floatX):
        """
            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights
            
            :type order: int32
            :param order: order of the RNN (used for higher order RNNs)
            
            :type n_in: int32
            :param n_in: number of input neurons
            
            :type n_hidden: int32
            :param n_hidden: number of hidden units
            
            :type dtype: theano.config.floatX
            :param dtype: theano 32/64bit mode
        """
                   
        #--------------------------------------------------------------------
        # PROBLEM section (needs review)
        #--------------------------------------------------------------------  
        """
         here the the recurrent delay elements (taps) will be created
         You can change the line below to add special delays like [-4,-1]
         
         Problem: single delays like tap = [-1] lead to a T.matrix() generation 
                  in step() line:117. A theano shape error will be generated. 
                  Hence we have to create at least a tuple [-1,0] for order.1 RNNs.
                  Tap[0] will not be used in step (skipped)
                      
         QuickFIX: tap = [-1,0] -> tap[0] won't be used
         Description: we have to create at least a tuple [-1,0] used in
                      outputs_info line:91
                      otherwise a T.matrix will be generated in step(),
                      leading to an theano shape error.
         Discuss: @theano-dev-group 
        """
        # order of the rnn
        self.order = order
        # initialize delay elements           
        self.taps = numpy.arange(order+1)-order                                    
        # input over the time (1st dim is the time)
        self.x  = T.matrix('x')            
        # target over the time (1st dim is the time)
        self.y  = T.ivector('y')  
        # recurrent activations over the time (1st dim is the time)
        self.H = T.matrix()                         
        # learning rate
        self.lr = T.fscalar()       
        
        # input to hidden layer weights
        self.W_in = theano.shared(numpy.asarray(rng.uniform(
                low  = - numpy.sqrt(6./(n_in+n_hidden)),
                high = numpy.sqrt(6./(n_in+n_hidden)),
                size = (n_in, n_hidden)), 
                dtype = dtype), name='W_in')                                  
        self.b_in = theano.shared(numpy.zeros((n_hidden,), dtype=dtype), name='b_in')
        
        # recurrent activations        
        self.h = numpy.zeros((n_hidden, n_hidden), dtype = dtype)
        # recurrent weights as real values
        self.W = [theano.shared(numpy.asarray(rng.uniform(low  = - numpy.sqrt(6./(n_in+n_hidden)),\
                                    high = numpy.sqrt(6./(n_in+n_hidden)),\
                                    size = (n_hidden, n_hidden)), dtype = dtype),\
                                    name='W_r'+str(u)) for u in range(self.order)]
                        
        # hidden to output layer weights
        self.W_out = theano.shared(numpy.asarray(rng.uniform(
                low  = - numpy.sqrt(6./(n_hidden+n_out)),
                high = numpy.sqrt(6./(n_hidden+n_out)),
                size = (n_hidden, n_out)), 
                dtype = dtype), name = 'W_out')
        self.b_out = theano.shared(numpy.zeros((n_out,), dtype=dtype), name='b_out')
                                            
        # stack the network parameters
        self.params = []
        self.params.extend(self.W)
        self.params.extend([self.W_in, self.b_in])
        
        # this is the recursive BBTP loop
        # `self.H` are the recurrent activations, 'y_act_list' is the output for the
        # entire sequence over the time
        (y_act_list, updates) = theano.scan(fn = self.step, \
                             sequences = dict(input = self.x, taps = [0]), \
                             outputs_info = [dict(initial = self.H, taps = list(self.taps)), None], \
                             non_sequences  = self.params)
                
        # compute the output signal (using a logistic regression)
        # Note: we only need the final outcomes, not intermediary values, hence y_act_list[-1]                                        
        self.y_act = T.nnet.softmax(T.dot(y_act_list[-1], self.W_out) + self.b_out)
        
        # error between output and target
        self.cost = -T.mean(T.log(self.y_act)[T.arange(self.y.shape[0]),self.y])        
                                                             
        # add the the network parameters of the output layer                    
        self.params.extend([self.W_out, self.b_out])
        
    #---------------------------------------------------------------------------------
    def step(self, u_t, *args):     
        """
            step function to calculate BPTT
            
            type u_t: T.matrix()
            param u_t: input sequence of the network
            
            type *args: python parameter list
            param *args: this is needed to implement a more general model of the step function
                         see theano@users: http://groups.google.com/group/theano-users/ \
                         browse_thread/thread/2fa44792c9cdd0d5
            
        """
        
        # get the recurrent activations
        # (+1 is needed for the shape, since we added t[0] in line:40, to
        # make self.order always at least a tuple [-1,0]
        # see comment line:40
        r_act_vals = [args[u] for u in xrange(self.order+1)]
        
        # get the recurrent weights
        r_weights = [args[u] for u in xrange(self.order+1, self.order*2+1)]           
                
        # get the input/output weights        
        W_in = args[self.order*2+1]
        b_in = args[self.order*2+2]
                
        # calculate the new activation  
        # we skip the t=0 activation          
        #act = [theano.dot(r_act_vals[u+1], r_weights[u]) for u in xrange(self.order)]              
        act = theano.dot(r_act_vals[0], r_weights[0])
        for u in xrange(1,self.order):   
            act += T.dot(r_act_vals[u], r_weights[u])
        
        
        #x_t = T.tanh(theano.dot(u_t, W_in) + b_in + T.sum(act,axis=0))
        x_t = T.tanh(T.dot(u_t, W_in) + b_in + act)    
                        
        return [x_t, x_t]
                
    #---------------------------------------------------------------------------------
    def build_finetune_functions(self, train_set_x, train_set_y, learning_rate, mode):
        """
            type train_set_x: T.matrix()
            param train_set_x: training input sequences of the network
            
            type train_set_y: T.ivector()
            param train_set_y: training output sequences of the network
            
            type learning_rate: float32
            param learning_rate: learning_rate of the training algorithm
            
            type mode: str
            param mode: theano function compile mode 
            
        """
                
        #-----------------------------------------
        # THEANO variables for the data access
        #-----------------------------------------    
        # we specify direct indizes here, since this allows to have different seq. lengths
        i_idx_0 = T.iscalar('input_start') # index to input start
        i_idx_1 = T.iscalar('input_stop') # index to input stop 
        t_idx_0 = T.iscalar('target_start') # index to target start
        t_idx_1 = T.iscalar('target_stop') # index to target stop
        
        #-----------------------------------------
        # THEANO train function
        #-----------------------------------------         
        gparams = []
        for param in self.params:
            gparam  = T.grad(self.cost, param)
            gparams.append(gparam)

        # specify how to update the parameters of the model as a dictionary
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - learning_rate*gparam 
        
        # compiling a Theano function `train_fn` that returns the cost, but  
        # in the same time updates the parameter of the model based on the rules 
        # defined in `updates`  
        t_inputs = [i_idx_0] + [i_idx_1] + [t_idx_0] + [t_idx_1] + [self.H]                                                  
        self.train_fn = theano.function(inputs = t_inputs, 
                outputs = self.cost, 
                updates = updates,                                        
                givens={self.x:train_set_x[i_idx_0:i_idx_1],
                        self.y:train_set_y[t_idx_0:t_idx_1]},
                mode = mode)
                                        
        # return function (defined as non theano function)
        return self.train

    #---------------------------------------------------------------------------------
    def train(self, i_idx_0, i_idx_1, t_idx_0, t_idx_1):
        
        error = self.train_fn(i_idx_0, i_idx_1, t_idx_0, t_idx_1, self.h) 

#---------------------------------------------------------------------------------
class PrintEverythingMode(theano.Mode):
    def __init__(self, linker, optimizer=None):
        def print_eval(i, node, fn):
            print i, node, [input[0] for input in fn.inputs],
            fn()
            print [output[0] for output in fn.outputs]
        wrap_linker = theano.gof.WrapLinkerMany([linker], [print_eval])
        super(PrintEverythingMode, self).__init__(wrap_linker, optimizer) 
 
#---------------------------------------------------------------------------------
class Engine(object):

    def __init__(self, 
                learning_rate = 0.01, 
                n_epochs = 20,
                hidden = 50,
                order = 1):
 
        #-----------------------------------------
        # BENCHMARK SETUP
        #-----------------------------------------    
        # setup mode   
        #profmode = 'FAST_RUN'      
        profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())        
        #profmode = PrintEverythingMode(theano.gof.OpWiseCLinker(), 'fast_run')
        
        # setup data
        N = 100 # number of sequences
        L = 5 # number of feature vectors in sequence
        n_in = 784 # number of number of features (input)
        n_out = 11 # number of features (target)
        data_x = numpy.random.randn(N*L, n_in)        
        data_y = numpy.random.randn(N*L, n_out).reshape(L*N*n_out)     
        
        # create the shared vars          
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))            
        shared_y = T.cast(theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX)), 'int32')                
        
        # build a sequence index lookup table 
        # (a nice thing to have if you're using variable seq. lengths and can be extended
        # to seq['inputs|targets']['train|test|valid'][idx] easily)        
        inputIndex = targetIndex = [ u*L for u in xrange(N+1)]                        
        seq = { 'inputs':inputIndex, 'targets':targetIndex }
                              
        #-----------------------------------------
        # RNN SETUP
        #-----------------------------------------           
        # initialize random generator                                                  
        rng = numpy.random.RandomState(1234)      
        # construct the CTC_RNN class
        classifier = RNN(rng=rng, order=order, n_in=n_in, n_hidden=hidden, n_out=n_out)    
        # fetch the training function
        train_fn = classifier.build_finetune_functions(shared_x, shared_y, learning_rate, mode=profmode)             
                             
        #-----------------------------------------
        # BENCHMARK START
        #-----------------------------------------
                                                    
        # start the benchmark         
        for _ in xrange(n_epochs) :
            start_time = time.clock()              
            [ train_fn(seq['inputs'][j], seq['inputs'][j+1], seq['targets'][j], seq['targets'][j+1]) \
                    for j in xrange(N) ]            
            print >> sys.stderr, ('     training epoch time (%.5fm)' % ((time.clock()-start_time)/60.))
        
        profmode.print_summary()

#---------------------------------------------------------------------------------    
if __name__ == '__main__':
    
    Engine()
    