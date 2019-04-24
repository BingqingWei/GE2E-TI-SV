__author__ = 'Bingqing Wei'
import keras
from models.base import *

class OLD_LSTM_Model(Model):
    def build_model(self, batch):
        with tf.variable_scope('lstm'):
            cells = [tf.nn.rnn_cell.LSTMCell(num_units=config.nb_hidden, num_proj=config.nb_proj,
                                             initializer=tf.initializers.glorot_normal)
                     for i in range(config.nb_layers)]
            lstm = tf.nn.rnn_cell.MultiRNNCell(cells)
            outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)
            embedded = outputs[-1]
            # shape = (N * M, nb_proj)
            embedded = normalize(embedded)
        return embedded

class LSTM_Model(Model):
    def build_model(self, batch):
        with tf.variable_scope('lstm'):
            cells = [tf.nn.rnn_cell.LSTMCell(num_units=config.nb_hidden,
                                             num_proj=config.nb_proj,
                                             initializer=tf.initializers.glorot_normal)
                     for _ in range(config.nb_layers)]
            lstm = tf.keras.layers.StackedRNNCells(cells)
            embedded = tf.keras.layers.RNN(cell=lstm, return_sequences=False,
                                           return_state=False, time_major=True)(batch)

            # shape = (N * M, nb_proj)
            embedded = normalize(embedded)
        return embedded

class GRU_Model(Model):
    """
    Using GRU
    If nb_layers > 1
    Then use nb_hidden for first nb_layers - 1 GRU layers
    Then use nb_proj for the last layer
    """

    def build_model(self, batch):
        inputs = batch
        for _ in range(config.nb_layers - 1):
            inputs = tf.keras.layers.GRU(units=config.nb_hidden, return_sequences=True, return_state=False,
                                         time_major=True, kernel_initializer=tf.initializers.glorot_normal)(inputs)
        x = tf.keras.layers.GRU(units=config.nb_proj, return_sequences=False, return_state=False,
                                time_major=True, kernel_initializer=tf.initializers.glorot_normal)(inputs)
        return normalize(x)


# provided by lawy623
class GRU_Dropout_Model(Model):
    def build_model(self, batch):
        with tf.variable_scope('gru-dropout'):
            cells = [tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(num_units=config.nb_hidden, kernel_initializer=tf.initializers.orthogonal,
                                       bias_initializer=tf.initializers.orthogonal), output_keep_prob=0.5)
                for _ in range(config.nb_layers-1)]
            cells.append(tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(num_units=config.nb_proj, kernel_initializer=tf.initializers.orthogonal,
                                       bias_initializer=tf.initializers.orthogonal), output_keep_prob=0.5))
            gru = tf.nn.rnn_cell.MultiRNNCell(cells)
            outputs, _ = tf.nn.dynamic_rnn(cell=gru, inputs=batch, dtype=tf.float32, time_major=True)
            embedded = outputs[-1]
            # shape = (N * M, nb_proj)
            embedded = normalize(embedded)
        return embedded

class LSTM_ATT_Model(Model):
    def build_model(self, batch):
        att_size = 128
        with tf.variable_scope('lstm-att'):
            cells = [tf.contrib.rnn.LSTMCell(num_units=config.nb_hidden, num_proj=config.nb_proj)
                     for i in range(config.nb_layers)]
            lstm = tf.contrib.rnn.MultiRNNCell(cells)
            outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)

            w_att = tf.Variable(tf.random.normal([config.nb_proj, att_size], stddev=0.1))
            b_att = tf.Variable(tf.random.normal([att_size], stddev=0.1))
            u_att = tf.Variable(tf.random.normal([att_size], stddev=0.1))
            v = tf.tanh(tf.tensordot(outputs, w_att, axes=1) + b_att)
            vu = tf.tensordot(v, u_att, axes=1)
            alphas = tf.nn.softmax(vu, axis=0, name='alphas')
            alphas = tf.expand_dims(tf.transpose(alphas, perm=[1,0]), -1)
            embedded = tf.reduce_sum(tf.transpose(outputs, perm = [1,0,2]) * alphas, 1)
            embedded = outputs[-1]

            # shape = (N * M, nb_proj)
            embedded = normalize(embedded)
        return embedded


class BI_LSTM_Model(Model):
    def build_model(self, batch):
        with tf.variable_scope('bi-lstm'):
            cells_fw = [tf.contrib.rnn.LSTMCell(num_units=config.nb_hidden, num_proj=config.nb_proj) for _ in range(config.nb_layers)]
            lstm_fw = tf.contrib.rnn.MultiRNNCell(cells_fw)
            cells_bw = [tf.contrib.rnn.LSTMCell(num_units=config.nb_hidden, num_proj=config.nb_proj) for _ in range(config.nb_layers)]
            lstm_bw = tf.contrib.rnn.MultiRNNCell(cells_bw)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw, cell_bw=lstm_bw, inputs=batch, dtype=tf.float32, time_major=True)
            outputs = tf.concat(outputs, 2)
            embedded = tf.math.add(outputs[0,:,:],outputs[-1,:,:]) / 2.0
            # shape = (N * M, 2*nb_proj)
            embedded = normalize(embedded)
        return embedded
