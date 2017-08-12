import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

# define a single LSTM cell with a 15D hidden vector
def lstm_cell():
	return rnn.BasicLSTMCell(15, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
	
x = tf.placeholder(tf.float32, [1, None, 20])

lstm0 = lstm_cell()
lstm1 = lstm_cell()
lstm2 = lstm_cell()

state_size0 = lstm0.state_size
state_size1 = lstm1.state_size
state_size2 = lstm2.state_size

c0 = tf.placeholder(tf.float32, [1, state_size0.c])
h0 = tf.placeholder(tf.float32, [1, state_size0.h])
c1 = tf.placeholder(tf.float32, [1, state_size1.c])
h1 = tf.placeholder(tf.float32, [1, state_size1.h])
c2 = tf.placeholder(tf.float32, [1, state_size2.c])
h2 = tf.placeholder(tf.float32, [1, state_size2.h])

step_size = tf.shape(x)[1:2]

state_in0 = rnn.LSTMStateTuple(c0, h0)
state_in1 = rnn.LSTMStateTuple(c1, h1)
state_in2 = rnn.LSTMStateTuple(c2, h2)

outputs0, state0 = tf.nn.dynamic_rnn(lstm0, x, initial_state=state_in0, sequence_length=step_size, time_major=False, scope='rnn0')
outputs0 = tf.reshape(outputs0, [1, -1, 15])

outputs1, state1 = tf.nn.dynamic_rnn(lstm1, outputs0, initial_state=state_in1, sequence_length=step_size, time_major=False, scope='rnn1')
outputs1 = tf.reshape(outputs1, [1, -1, 15])

outputs2, state2 = tf.nn.dynamic_rnn(lstm2, outputs1, initial_state=state_in2, sequence_length=step_size, time_major=False, scope='rnn2')
outputs2 = tf.reshape(outputs2, [-1, 15])

# Initialise the graph
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# Initialise the data
xx = np.random.rand(5, 20)
c0_init = np.zeros((1, state_size0.c), np.float32)
h0_init = np.zeros((1, state_size0.h), np.float32)
c1_init = np.zeros((1, state_size1.c), np.float32)
h1_init = np.zeros((1, state_size1.h), np.float32)
c2_init = np.zeros((1, state_size2.c), np.float32)
h2_init = np.zeros((1, state_size2.h), np.float32)

# run the experiment
pi = sess.run(outputs2, {x : [xx],  c0: c0_init, c1 : c1_init, c2 : c2_init, h0 : h0_init, h1 : h1_init, h2 : h2_init})
