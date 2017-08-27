import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
# use_tf100_api is a boolean flag : True means tf version is greater than 1.0.0
# I am using the tf version 1.2.1
use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')
openai = True


############################################################################################
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
	# x is a tf.Variable. x.get_shape returns tf dimension type. then as_list change it to the python list type.
	# [1:] takes the dimensions except the first one which is the batch size. 
	# In short, this function casts [batchsize, x, y] to [batchsize, x * y] image pixels to a vector.
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
	# a FC layer without any activation function
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
	# d is the length of the vector logits
	# tf.multinomial will firstly take the softmax of the logits vector to have a distribution, 
	# then sample according to this distribution.
	# logits - max(logits) is a computational trick. It can avoid the extreme large value of exp(.) but has no influence on the final result.
	# tf.squeeze casts [[1]] to [1]
	# in short, this function sample an one-hot action vector according to the pi(a | s) distribution
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)

def argmax_sample(logits, d):
    return tf.one_hot(np.argmax(logits), d)

############################################################################################
class LSTMPolicy_alpha(object):
    def __init__(self, ob_space, ac_space):
    	# ob_space is the dimension of the observation pixels. ac_space is the action space dimension
    	# x is the input images with dimension [batchsize, observation dimension]
    	# Pyhton syntax : a = b = 1. a and b have no reference or relationship.
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        if openai:
            for i in range(4):
                x = tf.nn.relu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
            x = tf.expand_dims(flatten(x), [0])
        else:
            conv1 = tf.contrib.layers.conv2d(x, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
            conv2 = tf.contrib.layers.conv2d(conv1, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2")
            fc1 = tf.contrib.layers.fully_connected(inputs=tf.contrib.layers.flatten(conv2), num_outputs=256, scope="fc1")
            x = tf.expand_dims(fc1, [0])
		
		# size of h, the hidden state vector
        size = 256
        self.size = size
        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

        # state_size has two fields: c and h
        self.state_size = lstm.state_size
        # step_size equals to the sequence size
        # Note : self.x is different from x. They have different dimensions
        step_size = tf.shape(self.x)[0:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]
        state_in = rnn.LSTMStateTuple(c_in, h_in)

        
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, x, initial_state=state_in, sequence_length=step_size, time_major=False)
        lstm_c, lstm_h = lstm_state

        x = tf.reshape(lstm_outputs, [-1, size])
        # logits is pi(a | s)
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        # vf is V(s)
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        # list of the input variables, used in the gradient calculation of the loss function
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]
        
############################################################################################
class LSTMPolicy_beta(object):
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        if openai:
            for i in range(4):
                x = tf.nn.relu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
            x = flatten(x)
        else:
            conv1 = tf.contrib.layers.conv2d(x, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
            conv2 = tf.contrib.layers.conv2d(conv1, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2")
            x = tf.contrib.layers.fully_connected(inputs=tf.contrib.layers.flatten(conv2), num_outputs=256, scope="fc1")
            
        size = 256
        self.size = size

        x = tf.expand_dims(x, [0])
        
        def lstm_cell():
            return rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        
        stacked_lstm = rnn.MultiRNNCell([lstm_cell() for _ in range(3)], state_is_tuple=True)
        state_size = stacked_lstm.state_size
        self.state_size = state_size
        step_size = tf.shape(self.x)[0:1]

        c0_init = np.zeros((1, state_size[0].c), np.float32)
        h0_init = np.zeros((1, state_size[0].h), np.float32)
        c1_init = np.zeros((1, state_size[1].c), np.float32)
        h1_init = np.zeros((1, state_size[1].h), np.float32)
        c2_init = np.zeros((1, state_size[2].c), np.float32)
        h2_init = np.zeros((1, state_size[2].h), np.float32)
        self.state_init = [np.reshape(np.stack([c0_init, c1_init, c2_init], axis = 0), (-1, size)),
                           np.reshape(np.stack([h0_init, h1_init, h2_init], axis = 0), (-1, size))]

        c0 = tf.placeholder(tf.float32, [1, state_size[0].c])
        h0 = tf.placeholder(tf.float32, [1, state_size[0].h])
        c1 = tf.placeholder(tf.float32, [1, state_size[1].c])
        h1 = tf.placeholder(tf.float32, [1, state_size[1].h])
        c2 = tf.placeholder(tf.float32, [1, state_size[2].c])
        h2 = tf.placeholder(tf.float32, [1, state_size[2].h])
        self.state_in = [[c0, c1, c2], [h0, h1, h2]]

        state_in = (rnn.LSTMStateTuple(c0, h0), rnn.LSTMStateTuple(c1, h1), rnn.LSTMStateTuple(c2, h2))
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(stacked_lstm, x, initial_state=state_in, sequence_length=step_size, time_major=False)

        self.lstm_c = tf.reshape(tf.stack(values = [lstm_state[0].c, lstm_state[1].c, lstm_state[2].c], axis = 0), (-1, size))      
        self.lstm_h = tf.reshape(tf.stack(values = [lstm_state[0].h, lstm_state[1].h, lstm_state[2].h], axis = 0), (-1, size))      
        x = tf.reshape(lstm_outputs, [-1, size])

        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [self.lstm_c, self.lstm_h]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, x, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [x], self.state_in[0][0]: c[0:1], self.state_in[0][1]: c[1:2], 
                        self.state_in[0][2]: c[2:3], self.state_in[1][0]: h[0:1], 
                        self.state_in[1][1]: h[1:2], self.state_in[1][2]: h[2:3]})

    def value(self, x, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [x], self.state_in[0][0]: c[0:1], self.state_in[0][1]: c[1:2], 
                        self.state_in[0][2]: c[2:3], self.state_in[1][0]: h[0:1], 
                        self.state_in[1][1]: h[1:2], self.state_in[1][2]: h[2:3]})[0]
                   
#############################################################################################################
class LSTMPolicy_gamma(object):
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        if openai:
            for i in range(4):
                x = tf.nn.relu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
            #x = tf.expand_dims(flatten(x), [0])
            x = flatten(x)
        else:
            conv1 = tf.contrib.layers.conv2d(x, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
            conv2 = tf.contrib.layers.conv2d(conv1, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2")
            x = tf.contrib.layers.fully_connected(inputs=tf.contrib.layers.flatten(conv2), num_outputs=256, scope="fc1")
            #x = tf.expand_dims(x, [0])
            
        size = 256
        self.size = size
        self.h_aux0 = tf.placeholder(tf.float32, [None, size])
        self.h_aux1 = tf.placeholder(tf.float32, [None, size])
        self.h_aux2 = tf.placeholder(tf.float32, [None, size])
        
        x = tf.concat([x, self.h_aux0], 1)
        x = tf.expand_dims(x, [0])
        
        lstm0 = rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        lstm1 = rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        lstm2 = rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

        state_size0 = lstm0.state_size
        state_size1 = lstm1.state_size
        state_size2 = lstm2.state_size

        step_size = tf.shape(self.x)[0:1]

        c0_init = np.zeros((1, state_size0.c), np.float32)
        h0_init = np.zeros((1, state_size0.h), np.float32)
        c1_init = np.zeros((1, state_size1.c), np.float32)
        h1_init = np.zeros((1, state_size1.h), np.float32)
        c2_init = np.zeros((1, state_size2.c), np.float32)
        h2_init = np.zeros((1, state_size2.h), np.float32)
        self.state_init = [[c0_init, c1_init, c2_init], [h0_init, h1_init, h2_init]]
        
        c0 = tf.placeholder(tf.float32, [1, state_size0.c])
        h0 = tf.placeholder(tf.float32, [1, state_size0.h])
        c1 = tf.placeholder(tf.float32, [1, state_size1.c])
        h1 = tf.placeholder(tf.float32, [1, state_size1.h])
        c2 = tf.placeholder(tf.float32, [1, state_size2.c])
        h2 = tf.placeholder(tf.float32, [1, state_size2.h])
        
        self.state_in = [[c0, c1, c2], [h0, h1, h2]]
        
        state_in0 = rnn.LSTMStateTuple(c0, h0)
        state_in1 = rnn.LSTMStateTuple(c1, h1)
        state_in2 = rnn.LSTMStateTuple(c2, h2)
        
        outputs0, state0 = tf.nn.dynamic_rnn(lstm0, x, initial_state=state_in0, sequence_length=step_size, time_major=False, scope='rnn0')
        outputs0 = tf.concat([tf.reshape(outputs0, [-1, size]), self.h_aux1], 1)
        outputs0 = tf.reshape(outputs0, [1, -1, size*2])
        
        outputs1, state1 = tf.nn.dynamic_rnn(lstm1, outputs0, initial_state=state_in1, sequence_length=step_size, time_major=False, scope='rnn1')
        outputs1 = tf.concat([tf.reshape(outputs1, [-1, size]), self.h_aux2], 1)
        outputs1 = tf.reshape(outputs1, [1, -1, size*2])
        
        outputs2, state2 = tf.nn.dynamic_rnn(lstm2, outputs1, initial_state=state_in2, sequence_length=step_size, time_major=False, scope='rnn2')
        x = tf.reshape(outputs2, [-1, size])
        
        self.lstm_c = [state0.c, state1.c, state2.c]    
        self.lstm_h = [state0.h, state1.h, state2.h]


        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [self.lstm_c, self.lstm_h]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, x, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [x], self.state_in[0][0]: c[0], self.state_in[0][1]: c[1], 
                        self.state_in[0][2]: c[2], self.state_in[1][0]: h[0], 
                        self.state_in[1][1]: h[1], self.state_in[1][2]: h[2], 
                        self.h_aux0: h[0], self.h_aux1: h[1], 
                        self.h_aux2: h[2]})

    def value(self, x, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [x], self.state_in[0][0]: c[0], self.state_in[0][1]: c[1], 
                        self.state_in[0][2]: c[2], self.state_in[1][0]: h[0], 
                        self.state_in[1][1]: h[1], self.state_in[1][2]: h[2], 
                        self.h_aux0: h[0], self.h_aux1: h[1], 
                        self.h_aux2: h[2]})[0]                   
                   
                   
############################################################################################################     
#class LSTMPolicy_gamma(object):
#    def __init__(self, ob_space, ac_space):
#        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
#        if openai:
#            for i in range(4):
#                x = tf.nn.relu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
#            x = tf.expand_dims(flatten(x), [0])
#        else:
#            conv1 = tf.contrib.layers.conv2d(x, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
#            conv2 = tf.contrib.layers.conv2d(conv1, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2")
#            fc1 = tf.contrib.layers.fully_connected(inputs=tf.contrib.layers.flatten(conv2), num_outputs=256, scope="fc1")
#            x = tf.expand_dims(fc1, [0])
#        
#        size = 256
#        def lstm_cell():
#            return rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
#        
#        stacked_lstm = rnn.MultiRNNCell([lstm_cell() for _ in range(3)], state_is_tuple=True)
#        state_size = stacked_lstm.state_size
#        self.state_size = state_size
#        step_size = tf.shape(self.x)[0]        

#        c0_init = np.zeros((1, state_size[0].c), np.float32)
#        h0_init = np.zeros((1, state_size[0].h), np.float32)
#        c1_init = np.zeros((1, state_size[1].c), np.float32)
#        h1_init = np.zeros((1, state_size[1].h), np.float32)
#        c2_init = np.zeros((1, state_size[2].c), np.float32)
#        h2_init = np.zeros((1, state_size[2].h), np.float32)
#        self.state_init = [np.reshape(np.stack([c0_init, c1_init, c2_init], axis = 0), (-1, size)),
#                           np.reshape(np.stack([h0_init, h1_init, h2_init], axis = 0), (-1, size))]

#        c0 = tf.placeholder(tf.float32, [1, state_size[0].c])
#        h0 = tf.placeholder(tf.float32, [1, state_size[0].h])
#        c1 = tf.placeholder(tf.float32, [1, state_size[1].c])
#        h1 = tf.placeholder(tf.float32, [1, state_size[1].h])
#        c2 = tf.placeholder(tf.float32, [1, state_size[2].c])
#        h2 = tf.placeholder(tf.float32, [1, state_size[2].h])
#        self.state_in = [[c0, c1, c2], [h0, h1, h2]]
        

#        H_init0 = tf.TensorArray(dtype=tf.float32, size=(step_size+1), clear_after_read = False)
#        H_init1 = tf.TensorArray(dtype=tf.float32, size=(step_size+1), clear_after_read = False)
#        H_init2 = tf.TensorArray(dtype=tf.float32, size=(step_size+1), clear_after_read = False)
#        C_init0 = tf.TensorArray(dtype=tf.float32, size=(step_size+1), clear_after_read = False)
#        C_init1 = tf.TensorArray(dtype=tf.float32, size=(step_size+1), clear_after_read = False)
#        C_init2 = tf.TensorArray(dtype=tf.float32, size=(step_size+1), clear_after_read = False)

#        H_init0 = H_init0.write(0, h0)
#        H_init1 = H_init1.write(0, h1)
#        H_init2 = H_init2.write(0, h2)
#        C_init0 = C_init0.write(0, c0)
#        C_init1 = C_init1.write(0, c1)
#        C_init2 = C_init2.write(0, c2)

#        output_init = tf.TensorArray(dtype=tf.float32, size=step_size)
        
#        def cond(i, *args):
#            return tf.less(i, step_size)

#        def body(i, output_, h0_, h1_, h2_, c0_, c1_, c2_):
#            state_in_temp = (rnn.LSTMStateTuple(c0_.gather([i])[0], h0_.gather([i])[0]), rnn.LSTMStateTuple(c1_.gather([i])[0], h1_.gather([i])[0]), rnn.LSTMStateTuple(c2_.gather([i])[0], h2_.gather([i])[0]))
#            (O, S) = stacked_lstm(x[:, i, :], state_in_temp)
#            output_ = output_.write(i, O)
#            c0_ = c0_.write(i+1, S[0].c)
#            h0_ = h0_.write(i+1, S[0].h)
#            c1_ = c1_.write(i+1, S[1].c)
#            h1_ = h1_.write(i+1, S[1].h)
#            c2_ = c2_.write(i+1, S[2].c)
#            h2_ = h2_.write(i+1, S[2].h)
#            return i+1, output_, h0_, h1_, h2_, c0_, c1_, c2_
	
#        i, output, hidden0, hidden1, hidden2, cell0, cell1, cell2 = tf.while_loop(cond, body, [0, output_init, H_init0, H_init1, H_init2, C_init0, C_init1, C_init2])
		
#        x = tf.reshape(output.stack(), (-1, size))
#        hidden0 = tf.reshape(hidden0.stack(), (-1, size))
#        hidden1 = tf.reshape(hidden1.stack(), (-1, size))
#        hidden2 = tf.reshape(hidden2.stack(), (-1, size))
#        cell0 = tf.reshape(cell0.stack(), (-1, size))
#        cell1 = tf.reshape(cell1.stack(), (-1, size))
#        cell2 = tf.reshape(cell2.stack(), (-1, size))
        
#        self.hidden0 = hidden0
#        self.hidden1 = hidden1
#        self.hidden2 = hidden2
#        self.lstm_c = tf.reshape(tf.stack(values = [cell0[-1], cell1[-1], cell2[-1]], axis = 0), (-1, size))      
#        self.lstm_h = tf.reshape(tf.stack(values = [hidden0[-1], hidden1[-1], hidden2[-1]], axis = 0), (-1, size))      

#        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
#        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
#        self.state_out = [self.lstm_c, self.lstm_h]
#        self.sample = categorical_sample(self.logits, ac_space)[0, :]
#        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

#    def get_initial_features(self):
#        return self.state_init

#    def act(self, x, c, h):
#        sess = tf.get_default_session()
#        return sess.run([self.sample, self.vf] + self.state_out,
#                        {self.x: [x], self.state_in[0][0]: c[0:1], self.state_in[0][1]: c[1:2], 
#                        self.state_in[0][2]: c[2:3], self.state_in[1][0]: h[0:1], 
#                        self.state_in[1][1]: h[1:2], self.state_in[1][2]: h[2:3]})

#    def value(self, x, c, h):
#        sess = tf.get_default_session()
#        return sess.run(self.vf, {self.x: [x], self.state_in[0][0]: c[0:1], self.state_in[0][1]: c[1:2], 
#                        self.state_in[0][2]: c[2:3], self.state_in[1][0]: h[0:1], 
#                        self.state_in[1][1]: h[1:2], self.state_in[1][2]: h[2:3]})[0]
