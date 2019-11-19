from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from md_lstm import multi_dimensional_rnn_while_loop

my_path = os.path.abspath(os.path.dirname(__file__))
os.environ['TF_ENABLE_COND_V2'] = '1'


class ModelValues:
    """
    Metrics for the model including batch size, image size and the maximum
    text label length.
    """
    batch_size = 50
    image_size = (128, 32)
    max_text_length = 32


class Model:
    """
    The model that gets implemented. Its CRNN with CTC-Loss calculation.
    """
    def __init__(self, char_list, multi_dimensional, must_restore=False):
        """
        Setup for the Model. Depending on the arguments, creates a 1 or 2D-LSTM
        :param char_list: the list of available characters
        :param must_restore: Boolean for restoring a saved model.
        """
        self.char_list = char_list
        self.must_restore = must_restore
        self.snap_id = 0

        self.is_train = tf.placeholder(tf.bool, name='is_train')

        self.input_imgs = tf.placeholder(tf.float32,
                                         shape=(None,
                                                ModelValues.image_size[0],
                                                ModelValues.image_size[1]))

        if multi_dimensional:
            print("Multi-dimensional LSTM")
            self.setup_md_lstm()
        else:
            print("Using one-dimensional LSTM")
            self.setup_od_puigcever()
        self.setup_ctc()

        self.batches_trained = 0
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)\
                .minimize(self.loss)

        (self.session, self.saver) = self.setup_tf()

    def setup_od_puigcever(self):
        """
        reconstruct the implementation from puigcerver. (1D-LSTM)
        """
        cnn_input = tf.expand_dims(input=self.input_imgs, axis = 3)

        kernel_values = [3, 3, 3, 3, 3]
        feature_values = [1, 16, 32, 48, 64, 80]
        layer_count = len(kernel_values)

        pool = cnn_input

        for i in range(layer_count):
            print(pool.shape)
            if i >= 2:
                pool = tf.nn.dropout(pool, rate=0.2)
            kernel = tf.Variable(tf.truncated_normal(
                [kernel_values[i], kernel_values[i], feature_values[i],
                 feature_values[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding="SAME", strides=(1, 1,
                                                                       1, 1))
            print(conv.shape)
            conv_norm = tf.layers.batch_normalization(conv,
                                                      training=self.is_train)
            pool = tf.nn.leaky_relu(conv_norm, alpha=0.01)
            if i <= 1:
                pool = tf.nn.max_pool(pool, (1, 2, 2, 1), (1, 2, 2, 1),
                                      "VALID")
            else:
                pool = tf.nn.max_pool(pool, (1, 1, 2, 1), (1, 1, 2, 1),
                                      "VALID")

        rnn_input = tf.squeeze(pool, axis=[2])

        numHidden = 256
        cells = [
            tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True)
            for _ in range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN

        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked,
                                                        cell_bw=stacked,
                                                        inputs=rnn_input,
                                                        dtype=rnn_input.dtype)

        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        kernel = tf.Variable(
            tf.truncated_normal([1, 1, numHidden * 2, len(self.char_list) + 1],
                                stddev=0.1))

        self.rnn_output = tf.squeeze(
            tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1,
                                padding='SAME'), axis=[2])

    def setup_od_lstm(self):
        """
        create CNN and rnn layers and return output of these layers
        """
        print("using od_lstm")
        cnnIn4d = tf.expand_dims(input=self.input_imgs, axis=3)

        # list of parameters for the layers
        kernelVals = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        numLayers = len(strideVals)
        # create layers
        pool = cnnIn4d  # input to first CNN layer

        for i in range(numLayers):
            kernel = tf.Variable(tf.truncated_normal(
                [kernelVals[i], kernelVals[i], featureVals[i],
                 featureVals[i + 1]], stddev=0.1))

            conv = tf.nn.conv2d(pool, kernel, padding='SAME',
                                strides=(1, 1, 1, 1))
            conv_norm = tf.layers.batch_normalization(conv,
                                                      training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1),
                                  (1, strideVals[i][0], strideVals[i][1], 1),
                                  'VALID')

        self.cnnOut4d = pool
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        # basic cells which is used to build RNN
        numHidden = 256
        cells = [
            tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True)
            for _ in range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN

        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked,
                                                        cell_bw=stacked,
                                                        inputs=rnnIn3d,
                                                        dtype=rnnIn3d.dtype)

        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(
            tf.truncated_normal([1, 1, numHidden * 2, len(self.char_list) + 1],
                                stddev=0.1))

        self.rnn_output = tf.squeeze(
            tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1,
                                padding='SAME'), axis=[2])

    def setup_md_lstm(self):
        """
        Sets up a multi dimensional lstm. The md-lstm implementation is taken
        from https://github.com/philipperemy/tensorflow-multi-dimensional-lstm
        :return: returns the output of the last LSTM for the decoder.
        """
        print("using md_lstm")
        cnn_input = tf.expand_dims(input=self.input_imgs, axis=3)
        pool = cnn_input

        kernel = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1))
        conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
        relu = tf.nn.leaky_relu(conv)
        drop = tf.nn.dropout(relu, rate=0.4)

        md_1, _ = multi_dimensional_rnn_while_loop(16, drop, sh=(1, 1),
                                                scope_n='layer1')

        kernel1 = tf.Variable(tf.truncated_normal([2, 4, 16, 32], stddev=0.1))
        conv2 = tf.nn.conv2d(md_1, kernel1, padding='SAME',
                             strides=(1, 2, 4, 1))
        relu2 = tf.nn.leaky_relu(conv2)
        drop2 = tf.nn.dropout(relu2, rate=0.4)

        md_2, _ = multi_dimensional_rnn_while_loop(32, drop2, sh=(1, 1),
                                                 scope_n='layer2')

        kernel2 = tf.Variable(tf.truncated_normal([2, 4, 32, 64], stddev=0.1))
        conv3 = tf.nn.conv2d(md_2, kernel2, padding='SAME',
                             strides=(1, 2, 4, 1))
        relu3 = tf.nn.leaky_relu(conv3)
        drop3 = tf.nn.dropout(relu3, rate=0.4)

        md_3, _ = multi_dimensional_rnn_while_loop(64, drop3, sh=(1, 1),
                                                   scope_n='layer3')

        kernel3 = tf.Variable(tf.truncated_normal([1, 2, 64, 128], stddev=0.1))
        conv4 = tf.nn.conv2d(md_3, kernel3, padding='SAME',
                             strides=(1, 1, 2, 1))
        relu4 = tf.nn.leaky_relu(conv4)

        rnn_input = tf.squeeze(relu4, axis=[2])

        num_hidden = 128
        cells = [tf.contrib.rnn.LSTMCell(num_units=num_hidden,
                                         state_is_tuple=True)
                 for _ in range(2)]
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked,
                                                        cell_bw=stacked,
                                                        inputs=rnn_input,
                                                        dtype=rnn_input.dtype)

        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        kernel = tf.Variable(tf.truncated_normal([1, 1, num_hidden * 2,
                                                  len(self.char_list) + 1],
                                                 stddev=0.1))
        self.rnn_output = tf.squeeze(tf.nn.atrous_conv2d(value=concat,
                                                         filters=kernel,
                                                         rate=1,
                                                         padding='SAME'),
                                     axis=[2])
        print(self.rnn_output.shape)

    def setup_ctc(self):
        """
        sets up the ctc_loss calculation.
        :return: returns the different losses and the used decoder.
        """
        self.ctc_input = tf.transpose(self.rnn_output, [1, 0, 2])
        self.gt_texts = tf.SparseTensor(tf.placeholder(tf.int64,
                                                       shape=[None, 2]),
                                        tf.placeholder(tf.int32, [None]),
                                        tf.placeholder(tf.int64, [2]))

        self.seq_length = tf.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gt_texts,
                                                  inputs=self.ctc_input,
                                                  sequence_length=
                                                  self.seq_length,
                                                  ctc_merge_repeated=True))
        self.saved_ctc_input = tf.placeholder(tf.float32,
                                              shape=
                                              [ModelValues.max_text_length,
                                               None, len(self.char_list)
                                               + 1])
        self.loss_per_element = tf.nn.ctc_loss(labels=self.gt_texts,
                                               inputs=self.saved_ctc_input,
                                               sequence_length=self.seq_length,
                                               ctc_merge_repeated=True)

        self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctc_input,
                                                     sequence_length=
                                                     self.seq_length,
                                                     beam_width=50,
                                                     merge_repeated=False)

    def setup_tf(self):
        """
        sets up the tensorflow session and restores a saved model, if
        must_restore is true.
        :return: returns the session and the saver
        """
        session = tf.Session()
        saver = tf.train.Saver(max_to_keep=1)
        model_directory = '../models/'
        latest_snapshot = tf.train.latest_checkpoint(model_directory)

        if self.must_restore and not latest_snapshot:
            raise Exception('No saved model found.')

        if latest_snapshot:
            saver.restore(session, latest_snapshot)
        else:
            session.run(tf.global_variables_initializer())

        return session, saver

    def to_sparse(self, texts):
        """
        Puts the ground truth texts into sparse tensors for the ctc_loss.
        :param texts: ground truth texts
        :return: returns the values for ctc: indices, values and shape
        """
        indices = []
        values = []
        shape = [len(texts), 0]

        for (batch_element, text) in enumerate(texts):
            label_string = [self.char_list.index(c) for c in text]
            if len(label_string) > shape[1]:
                shape[1] = len(label_string)
            for (i, label) in enumerate(label_string):
                indices.append([batch_element, i])
                values.append(label)
        return indices, values, shape

    def decoder_output_to_text(self, ctc_output, batch_size):
        """
        Changes the output to text.
        :param ctc_output: output from the ctc_decoder
        :param batch_size: batch size
        :return: returns s lidz of word strings
        """
        encoded_label_strings = [[] for i in range(batch_size)]

        decoded = ctc_output[0][0]

        idx_dict = {b: [] for b in range(batch_size)}
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batch_element = idx2d[0]
            encoded_label_strings[batch_element].append(label)

        return [str().join([self.char_list[c] for c in label_string])
                for label_string in encoded_label_strings]

    def train_batch(self, batch):
        """
        trains the model on a batch.
        :param batch: the batch element from the loader.
        :return: returns the loss value of the batch
        """
        num_batch_elements = len(batch.images)
        sparse = self.to_sparse(batch.gt_texts)
        rate = 1.0 if self.batches_trained < 10 else \
            (1.0 if self.batches_trained < 10000 else 0.01)
        eval_list = [self.optimizer, self.loss]
        feed_dict = {self.input_imgs: batch.images, self.gt_texts: sparse,
                     self.seq_length:
                         [ModelValues.max_text_length] * num_batch_elements,
                     self.learning_rate: rate, self.is_train: True}
        (_, loss_value) = self.session.run(eval_list, feed_dict)
        self.batches_trained += 1
        return loss_value

    def infer_batch(self, batch, calc_probability=False,
                    probability_of_gt=False):
        """
        feeds a batch into the model to recognize text
        :param batch: the batch element from the loader
        :param calc_probability: if true, computes labeling probability
        :param probability_of_gt:
        :return: returns a list of words and the probability
        """
        num_batch_elements = len(batch.images)
        eval_rnn_output = calc_probability
        eval_list = [self.decoder] + ([self.ctc_input]
                                      if eval_rnn_output else [])
        feed_dict = {self.input_imgs: batch.images, self.seq_length:
                     [ModelValues.max_text_length] * num_batch_elements,
                     self.is_train: False}
        eval_res = self.session.run(eval_list, feed_dict)
        decoded = eval_res[0]
        texts = self.decoder_output_to_text(decoded, num_batch_elements)

        probs = None
        if calc_probability:
            sparse = self.to_sparse(batch.gt_texts) \
                if probability_of_gt else self.to_sparse(texts)
            ctc_input = eval_res[1]
            eval_list = self.loss_per_element
            feed_dict = {self.saved_ctc_input: ctc_input,
                         self.gt_texts: sparse,
                         self.seq_length: [ModelValues.max_text_length] *
                         num_batch_elements, self.is_train: False}
            loss_vals = self.session.run(eval_list, feed_dict)
            probs = np.exp(-loss_vals)

        return texts, probs

    def save(self):
        """
        Saves the model.
        """
        self.snap_id += 1
        model_path = os.path.join(my_path, '../model/snapshot')
        self.saver.save(self.session, model_path,
                        global_step=self.snap_id)

