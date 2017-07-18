import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
# use_tf100_api is a boolean flag : True means tf version is greater than 1.0.0
# I am using the tf version 1.2.1
use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')

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


############################################################################################
class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space):
    	# ob_space is the dimension of the observation pixels. ac_space is the action space dimension
    	# x is the input images with dimension [batchsize, observation dimension]
    	# Pyhton syntax : a = b = 1. a and b have no reference or relationship.
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

		# 4 layers of CNN 
		# tf.nn.elu means Exponential Linear Units. Like Sigmoid and RELU, it is a kind of activation function
        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
            
        # tf.expand_dims inserts a dimension of 1 into a tensor's shape
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(flatten(x), [0])
		
		# size of h, the hidden state vector
        size = 256
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
        self.h_in = h_in
        self.state_in = [c_in, h_in]

        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, x, initial_state=state_in, sequence_length=step_size, time_major=False)
        # the dim of lstm_c and lstm_h ?
        lstm_c, lstm_h = lstm_state
        self.h_out = lstm_h
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
    	# ob_space is the dimension of the observation pixels. ac_space is the action space dimension
    	# x is the input images with dimension [sequence size, observation dimension]. Typically [20, 42, 42, 1]
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

		# 4 layers of CNN 
		# tf.nn.elu means Exponential Linear Units. Like Sigmoid and RELU, it is a kind of activation function
        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
            
        # tf.expand_dims inserts a dimension of 1 into a tensor's shape
        # introduce a "fake" batch dimension of 1 after flatten
        # so that shape of x becomes [batchsize=1, sequence size = None, input size = pixel number]
        x = tf.expand_dims(flatten(x), [0])
		
		# size of h, the hidden state vector
        size = 256
        if use_tf100_api:
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        
        # state_size has two fields: c and h. In fact state_size.c = state_size.h = size
        self.state_size = lstm.state_size
        # step_size equals to the batch size
        step_size = tf.shape(self.x)[1:2]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, x, initial_state=state_in, sequence_length=step_size, time_major=False)
        # the dim of lstm_c and lstm_h ?
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
