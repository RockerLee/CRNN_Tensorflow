import tensorflow as tf
import numpy as np
import config
import utils
import os
from tensorflow.contrib import rnn

class CRNN(object):
    def __init__(self, batch_size, model_path, max_image_width, restore, debug, phase):
        self.step = 0
        self.__restore = restore
        self.__model_path = model_path
        self.__session = tf.Session()
        self.__debug = debug
        self.__batch_size = batch_size
        self.__save_path = os.path.join(model_path, 'ckp')
        self.__phase = phase
        
        # Building model
        with self.__session.as_default():
            (
                self.__inputs,
                self.__targets,
                self.__seq_len,
                self.__rnn_out,
                self.__decoded,
                self.__optimizer,
                self.__dist,
                self.__cost,
                self.__max_char_count,
                self.__global_step,
                self.__init,
                self.__learning_rate
            ) = self.crnn(max_image_width, batch_size)
            self.__init.run()
            
        #summary writer
        os.makedirs(config.PATH_TBOARD + '/train', exist_ok=True)
        os.makedirs(config.PATH_TBOARD + '/test', exist_ok=True)
        self.__train_writer = tf.summary.FileWriter(config.PATH_TBOARD + '/train')
        self.__test_writer = tf.summary.FileWriter(config.PATH_TBOARD + '/test')
        
        with self.__session.as_default():
            self.__saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            # Loading last save if needed
            if self.__restore:
                print('Restoring')
                ckpt = tf.train.latest_checkpoint(self.__model_path)
                if ckpt:
                    print('Checkpoint is valid')
                    self.step = int(ckpt.split('-')[1])
                    self.__saver.restore(self.__session, ckpt)
                    
    def crnn(self, max_width, batch_size):
        
        def BidirectionnalRNN(inputs, seq_len):
            """
                Bidirectionnal LSTM Recurrent Neural Network
            """
            with tf.variable_scope(None, default_name="bidirectional-rnn-1"):
                # Forward
                lstm_fw_cell_1 = rnn.BasicLSTMCell(256)
                # Backward
                lstm_bw_cell_1 = rnn.BasicLSTMCell(256)

                inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, seq_len, dtype=tf.float32)
                inter_output = tf.concat(inter_output, 2)

            with tf.variable_scope(None, default_name="bidirectional-rnn-2"):
                # Forward
                lstm_fw_cell_2 = rnn.BasicLSTMCell(256)
                # Backward
                lstm_bw_cell_2 = rnn.BasicLSTMCell(256)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, inter_output, seq_len, dtype=tf.float32)
                outputs = tf.concat(outputs, 2)
            return outputs
        
        def CNN(inputs):
            # input: [batch, height, width, 1], [batch, 32, 100, 1]
            # [batch, 32, 100, 64]
            conv1 = tf.layers.conv2d(inputs=inputs, filters = 64, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
            # [batch, 16, 50, 64]
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            # [batch, 16, 50, 128]
            conv2 = tf.layers.conv2d(inputs=pool1, filters = 128, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
            # [batch, 8, 25, 128]
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            # [batch, 8, 25, 256]
            conv3 = tf.layers.conv2d(inputs=pool2, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
            # batch_normalization
            bnorm1 = tf.layers.batch_normalization(conv3)
            # [batch, 8, 25, 256]
            conv4 = tf.layers.conv2d(inputs=bnorm1, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
            # [batch, 4, 25, 256]
            pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 1], strides=[2, 1], padding="valid")
            # [batch, 4, 25, 512]
            conv5 = tf.layers.conv2d(inputs=pool3, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
            # Batch normalization layer
            bnorm2 = tf.layers.batch_normalization(conv5)
            # [batch, 4, 25, 512]
            conv6 = tf.layers.conv2d(inputs=bnorm2, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
            # [batch, 2, 25, 512]
            pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 1], strides=[2, 1], padding="valid")
            # [batch, 1, 25, 512]
            conv7 = tf.layers.conv2d(inputs=pool4, filters = 512, kernel_size = (2, 2), strides=[2, 1], padding = "same", activation=tf.nn.relu)
            return conv7
        
        inputs = tf.placeholder(tf.float32, [batch_size, 32, max_width, 1])

        # Our target output
        targets = tf.sparse_placeholder(tf.int32, name='targets')
        # The length of the sequence
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        # CNN------------------------------------------------------------------------------
        cnn_output = CNN(inputs) #[batch, 1, 25, 512]
        if self.__debug:
            print('cnn_output:', cnn_output)

        reshaped_cnn_output = tf.reshape(cnn_output, [batch_size, -1, 512]) #[batch, 25, 512]
        max_char_count = reshaped_cnn_output.get_shape().as_list()[1] #25

        # RNN(LSTM)------------------------------------------------------------------------------
        rnn_output = BidirectionnalRNN(reshaped_cnn_output, seq_len) #[batch, seq_len, 512]
        if self.__debug:
            print('rnn_output:', rnn_output)
        if self.__phase == 'train':
            rnn_output = tf.nn.dropout(rnn_output, keep_prob=0.5)
            print('dropout: phase is train, has dropout')

        logits = tf.reshape(rnn_output, [-1, 512])

        W = tf.Variable(tf.truncated_normal([512, config.NUM_CLASSES], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[config.NUM_CLASSES]), name="b")
        logits = tf.matmul(logits, W) + b
        logits = tf.reshape(logits, [batch_size, -1, config.NUM_CLASSES]) #[batch, seq_len, NUM_CLASSES]
        # Final layer, the output of the BLSTM
        logits = tf.transpose(logits, (1, 0, 2)) #[seq_len, batch, NUM_CLASSES]

        # CTC------------------------------------------------------------------------------
        # Loss and cost calculation
        loss = tf.nn.ctc_loss(targets, logits, seq_len)

        cost = tf.reduce_mean(loss)

        # Training step
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(config.INITIAL_LEARNING_RATE, global_step,
                                               config.DECAY_STEPS, config.DECAY_RATE,
                                               staircase=config.STAIRCASE)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # The decoded answer
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
        if self.__debug:
            print('decoded:', decoded)

        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
        if self.__debug:
            print('dense_decoded:', dense_decoded)

        dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

        init = tf.global_variables_initializer()
        
        return inputs, targets, seq_len, logits, dense_decoded, optimizer, dist, cost, max_char_count, global_step, init, learning_rate
    
    def train(self):
        train_batches = utils.load_train_batches()
        
        with self.__session.as_default():
            print('Start train')
            self.__train_writer.add_graph(self.__session.graph)
            for i in range(self.step, config.EPOCHS):
                if self.__debug:
                    print('trining step ' + str(i) + ':----------------------------------------------------')
                iter_loss = 0
                dists = []
                learning_rates = []
                for batch_y, batch_dt, batch_x in train_batches:
                    # if self.__debug:
                        # print('next batch:')
                        # print('batch_y.shape: ', batch_y.shape)
                        # print(batch_dt)
                        # print(batch_x.shape)
                    op, decoded, loss_value, dist, learning_rate = self.__session.run(
                        [self.__optimizer, self.__decoded, self.__cost, self.__dist, self.__learning_rate],
                        feed_dict={
                            self.__inputs: batch_x,
                            self.__seq_len: [self.__max_char_count] * self.__batch_size,
                            self.__targets: batch_dt,
                            self.__global_step: self.step
                        }
                    )

                    if i % config.REPORT_STEPS == 0:
                        for j in range(2):
                            print('batch_y['+str(j)+']:', batch_y[j])
                            print('ground_truth_to_word:', utils.ground_truth_to_word(decoded[j]))

                    iter_loss += loss_value
                    dists.append(dist)
                    learning_rates.append(learning_rate)

                self.__saver.save(
                    self.__session,
                    self.__save_path,
                    global_step=self.step
                )
                dist = np.mean(dists)
                rate = np.mean(learning_rates)
                summary = tf.Summary()
                summary.value.add(tag="Edit Distance", simple_value=dist)
                summary.value.add(tag="Learning Rate", simple_value=rate)
                summary.value.add(tag="Loss", simple_value=iter_loss)
                self.__train_writer.add_summary(summary=summary, global_step=i)

                print('[{}] Iteration loss: {}, edit distance: {}----------'.format(self.step, iter_loss, dist))

                self.step += 1
    def test(self):
        print('Start test')