import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

# define a single LSTM cell with a 15D hidden vector
def lstm_cell():
	return rnn.BasicLSTMCell(15, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

# the height of the stacked LSTM is 3
num_layers = 3

# x is a placeholder for the input data with batchsize 1, dynamic sequence length and input data size 20
x = tf.placeholder(tf.float32, [1, None, 20])
# define a stacked LSTM network
stacked_lstm = rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
state_size = stacked_lstm.state_size

c0 = tf.placeholder(tf.float32, [1, state_size[0].c])
h0 = tf.placeholder(tf.float32, [1, state_size[0].h])
c1 = tf.placeholder(tf.float32, [1, state_size[1].c])
h1 = tf.placeholder(tf.float32, [1, state_size[1].h])
c2 = tf.placeholder(tf.float32, [1, state_size[2].c])
h2 = tf.placeholder(tf.float32, [1, state_size[2].h])
state_in = (rnn.LSTMStateTuple(c0, h0), rnn.LSTMStateTuple(c1, h1), rnn.LSTMStateTuple(c2, h2))

step_size = tf.shape(x)[1]

H_init0 = tf.TensorArray(dtype=tf.float32, size=(step_size+1), clear_after_read = False)
H_init1 = tf.TensorArray(dtype=tf.float32, size=(step_size+1), clear_after_read = False)
H_init2 = tf.TensorArray(dtype=tf.float32, size=(step_size+1), clear_after_read = False)
C_init0 = tf.TensorArray(dtype=tf.float32, size=(step_size+1))
C_init1 = tf.TensorArray(dtype=tf.float32, size=(step_size+1))
C_init2 = tf.TensorArray(dtype=tf.float32, size=(step_size+1))

H_init0 = H_init0.write(0, h0)
H_init1 = H_init1.write(0, h1)
H_init2 = H_init2.write(0, h2)
C_init0 = C_init0.write(0, c0)
C_init1 = C_init1.write(0, c1)
C_init2 = C_init2.write(0, c2)

output_init = tf.TensorArray(dtype=tf.float32, size=step_size)

def cond(i, *args):
    return tf.less(i, step_size)

def body(i, output_, h0_, h1_, h2_, c0_, c1_, c2_):
    state_in_temp = (rnn.LSTMStateTuple(c0_.gather([i])[0], h0_.gather([i])[0]), rnn.LSTMStateTuple(c1_.gather([i])[0], h1_.gather([i])[0]), rnn.LSTMStateTuple(c2_.gather([i])[0], h2_.gather([i])[0]))
    (O, S) = stacked_lstm(x[:, i, :], state_in_temp)
    output_ = output_.write(i, O)
    c0_ = c0_.write(i+1, S[0].c)
    h0_ = h0_.write(i+1, S[0].h)
    c1_ = c1_.write(i+1, S[1].c)
    h1_ = h1_.write(i+1, S[1].h)
    c2_ = c2_.write(i+1, S[2].c)
    h2_ = h2_.write(i+1, S[2].h)
    return i+1, output_, h0_, h1_, h2_, c0_, c1_, c2_
	
i, output, hidden0, hidden1, hidden2, cell0, cell1, cell2 = tf.while_loop(cond, body, [0, output_init, H_init0, H_init1, H_init2, C_init0, C_init1, C_init2])
		
output = output.stack()
hidden0 = hidden0.stack()
hidden1 = hidden1.stack()
hidden2 = hidden2.stack()

# Initialise the graph
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# Initialise the data
xx = np.random.rand(5, 20)
c0_init = np.zeros((1, state_size[0].c), np.float32)
h0_init = np.zeros((1, state_size[0].h), np.float32)
c1_init = np.zeros((1, state_size[1].c), np.float32)
h1_init = np.zeros((1, state_size[1].h), np.float32)
c2_init = np.zeros((1, state_size[2].c), np.float32)
h2_init = np.zeros((1, state_size[2].h), np.float32)

# run the experiment
pi, hh0, hh1, hh2 = sess.run([output, hidden0, hidden1, hidden2], {x : [xx], state_in[0].c : c0_init, state_in[1].c : c1_init, state_in[2].c : c2_init, state_in[0].h : h0_init, state_in[1].h : h1_init, state_in[2].h : h2_init})

