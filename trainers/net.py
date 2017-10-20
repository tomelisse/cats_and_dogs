import tensorflow as tf

class Network(object):
    ''' network '''
    def __init__(self, net_name, params):
        self.net_name = net_name
        self.params   = params

        # learning options
        self.n_epochs        = 10
        self.batch_size      = 5
        self.batch_per_epoch = 2
        self.learning_rate   = 0.1
        # the probability that the element is kept
        self.keep_prob = 0.8

        # plot data containers
        self.losses = []
        self.acces  = []

        # name params
        self.make_params(params)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.make_graph()
            self.prepare_training()

    def make_params(params):
        ''' name params '''
        self.width       = params[0]
        self.height      = params[1]
        self.depth       = params[2]
        self.n_classes   = params[3]
        self.n_filters   = params[4]
        self.filter_size = params[5]
        self.pool_size   = params[6]

    def make_graph(self):
        ''' define net structure '''
        in_shape = [None, self.widht, self.height, self.depth]
        self.inputs = tf.placeholder(tf.float32, shape = in_shape)

        out_shape = [None, self.n_classes]
        self.labels = tf.placeholder(tf.float32, shape = out_shape)

        # go through convolution
        self.conved = self.make_conv()
        # go through the fully-connected layer
        self.fced= self.make_fc()

        self.predictions = []

        # discrepancy between desired and actual output
        # both labels and outputs have to be of size batch_size x n_classes
        self.loss= tf.softmax_cross_entrophy(self.labels, self.output)
        correct = tf.argmax(self.labels, 1) == tf.argmax(self.output, 1)
        correct = tf.cast(correct, 'float')
        self.acc = tf.reduce_mean(correct)

    def make_conv(self):
        ''' convolutional layer '''
        # weights
        Wc_shape = [self.filter_size, 
                    self.filter_size, 
                    self.depth,
                    self.n_filters]
        Wc = tf.Variable(tf.truncated_normal(Wc_shape))

        # biases
        bc_shape = [self.n_filters]
        bc = tf.Variable(tf.truncated_normal(bc_shape))

        # apply convolution
        conved = tf.nn.conv2d(self.inputs, 
                            Wc, 
                            strides = [1,1,1,1], 
                            padding = 'SAME')
        # add bias
        biased = tf.nn.bias_add(conved, bc)

        # rectified linear unit
        relued = tf.nn.relu(biased)

        # down-sampling
        pool_window = [1, self.k_pool,
                       self.k_pool,1]
        pooled = tf.nn.max_pool(relued,
                                k_size  = pool_window,
                                strides = pool_window,
                                padding = 'SAME')

        return pooled

    def make_fc(self):
        ''' fully-connected layer '''
        # new width
        width = self.n_filters*self.width*self.height/(self.pool_size*self.pool_size)
        # fc input reshaping
        x = tf.reshape(self.conved, [-1, width])

        # weights
        Wfc_shape = [width, self.n_classes]
        Wfc = tf.Variable(tf.trunkated_normal(Wfc_shape))

        # biases
        bfc_shape = [self.n_classes]
        bfc = tf.Variable(tf.trunkated_normal(bfc_shape))

        # fc layer
        fced   = tf.nn.add_bias(tf.matmul(x, Wfc), bfc)
        # rectified linear unit
        relued = tf.nn.relu(fced) 
        
        # dropout
        dropped = tf.dropout(relued, self.keep_prob)

        return dropped

    def prepare_training(self):
        ''' define gradient update method '''
        optimizer            = tf.train.AdamOptimizer(self.learning_rate)
        self.gradient_update = optimizer.minimize(self.loss)

    def train(self, dataset):
        ''' train the nextwork '''
        with tf.Session(graph = self.graph) as sess:
            # initialize all variables
            sess.run(tf.global_variables_initializers)
            for epoch in range(self.n_epochs):
                # x - input, y - desired output
                x, y = dataset.next_batch(self.batch_size)
                # resize x and y if necessary
                # define data to place in the placeholders
                fd = {self.input = x, self.labels = y}
                # define the operations to be run
                operations = [self.gradient_update, self.loss, self.acc]

                # perform the STEP
                _, loss, acc = sess.run(operations, feed_dict = fd)

                # plots data
                self.losses.append(loss)
                self.acces.append(acc)
                
