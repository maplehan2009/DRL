import tensorflow as tf
import numpy

class my_cnn_agent:
    def __init__(self, n_action, shape_state, BATCH_SIZE):
        self.train_data_node, self.logits, self.conv1_weights, self.conv1_biases,\
              self.conv2_weights, self.conv2_biases, self.fc1_weights, self.fc1_biases,\
                      self.fc2_weights, self.fc2_biases = self.createnn(n_action, shape_state, BATCH_SIZE)
        self.createtrainingmethod(BATCH_SIZE)
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
      
    def createnn(self, n_action, shape_state, BATCH_SIZE):
        image_size0, image_size1, num_channels = shape_state
        train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, image_size0, image_size1, num_channels))
        conv1_weights = tf.Variable(
              tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                                  stddev=0.1,
                                  seed=SEED, dtype=tf.float32))
        conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
        conv2_weights = tf.Variable(tf.truncated_normal(
              [5, 5, 32, 64], stddev=0.1,
              seed=SEED, dtype=tf.float32))
        conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
        fc1_weights = tf.Variable(  # fully connected, depth 512.
              tf.truncated_normal([image_size0 // 4 * image_size1 // 4 * 64, 512],
                                  stddev=0.1,
                                  seed=SEED,
                                  dtype=tf.float32))
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
        fc2_weights = tf.Variable(tf.truncated_normal([512, n_action],
                                                        stddev=0.1,
                                                        seed=SEED,
                                                        dtype=tf.float32))
        fc2_biases = tf.Variable(tf.constant(
              0.1, shape=[n_action], dtype=tf.float32))
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv1 = tf.nn.conv2d(train_data_node,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool1 = tf.nn.max_pool(relu1,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        conv2 = tf.nn.conv2d(pool1,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # activations such that no rescaling is needed at evaluation time.
        logits = tf.matmul(hidden, fc2_weights) + fc2_biases
        return train_data_node, logits, conv1_weights, conv1_biases, conv2_weights, \
                conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases
                            
    def createtrainingmethod(self, BATCH_SIZE):
    	#TODO: modify the code here to adapt to Q learning loss function
        self.train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.train_labels_node, logits=self.logits))
        self.batch = tf.Variable(0, dtype=tf.float32)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        self.learning_rate = 0.01
        # Use simple momentum for the optimization.
        self.optimizer = tf.train.Optimizer(self.learning_rate, 0.9).minimize(self.loss, global_step=self.batch) 
        
    def train_nn(self, batch_data, batch_labels):
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {self.train_data_node: batch_data,
                   self.train_labels_node: batch_labels}
        # Run the optimizer to update weights.
        self.session.run(self.optimizer, feed_dict=feed_dict)
        
    def inference(self, data):
        """Get all predictions for a dataset by running it in small batches."""
        predictions = self.session.run(
                tf.nn.softmax(self.logits),
                feed_dict={self.train_data_node: data})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
        return np.argmax(predictions)
