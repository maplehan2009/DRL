import tensorflow as tf
import numpy as np

SEED = 66478

class my_cnn_agent:
    def __init__(self, n_action, shape_state, BATCH_SIZE):
        self.n_action = n_action
        self.BATCH_SIZE = BATCH_SIZE
        self.memory = []
        self.train_data_node, self.logits, self.fc1_weights, self.fc1_biases,\
                      self.fc2_weights, self.fc2_biases = self.createnn(shape_state)
        self.createtrainingmethod()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
      
    def createnn(self, shape_state):
        image_size0, image_size1, _ = shape_state
        train_data_node = tf.placeholder(tf.float32, shape=(None, image_size0, image_size1))
        fc1_weights = tf.Variable(  # fully connected, depth 512.
              tf.truncated_normal([image_size0 * image_size1, 512],
                                  stddev=0.1,
                                  seed=SEED,
                                  dtype=tf.float32))
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
        fc2_weights = tf.Variable(tf.truncated_normal([512, self.n_action],
                                                        stddev=0.1,
                                                        seed=SEED,
                                                        dtype=tf.float32))
        fc2_biases = tf.Variable(tf.constant(0.1, shape=[self.n_action], dtype=tf.float32))
        
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        reshape = tf.reshape(train_data_node, [-1, image_size0 * image_size1])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # activations such that no rescaling is needed at evaluation time.
        logits = tf.matmul(hidden, fc2_weights) + fc2_biases
        return train_data_node, logits, fc1_weights, fc1_biases, fc2_weights, fc2_biases
                            
    def createtrainingmethod(self):
        self.actionvector = tf.placeholder("float", [None, self.n_action])
        self.Gt = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.actionvector, self.logits), reduction_indices = 1)
        self.loss = tf.reduce_mean(tf.square(self.Gt - Q_action))
        self.trainStep = tf.train.RMSPropOptimizer(0.0025, 0.99, 0.0, 1e-6).minimize(self.loss)
        
    def train_my_agent(self):
        l = list(np.random.choice(len(self.memory), self.BATCH_SIZE, replace=False))
        minibatch = []
        for i in l:
            minibatch.append(self.memory[i])
            
        s = [data[0] for data in minibatch]
        a = [data[1] for data in minibatch]
        r = [data[2] for data in minibatch]
        s2 = [data[3] for data in minibatch]
        
        Gt = []
        Q2 = self.logits.eval(feed_dict = {self.train_data_node : s2})
        for i in range(self.BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                Gt.append(r[i])
            else:
                Gt.append(r[i] + 0.9 * np.max(Q2[i]))
        
        self.trainStep.run(feed_dict = {self.Gt : Gt, self.actionvector : a, self.train_data_node : s})
        
    def inference(self, current_state):
        QValue = self.logits.eval(feed_dict = {self.train_data_node : [current_state]})
        epsilon = 0.3
        if np.random.rand() > epsilon:
        	action = np.argmax(QValue)
        else:
        	action = int(np.random.rand() * self.n_action)
        return action
    
    def get_memory(self, s, a, r, s2, terminal):
        aa = [0] * self.n_action
        aa[a] = 1
        self.memory.append((s, aa, r, s2, terminal))
        
