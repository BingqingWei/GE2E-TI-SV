__author__ = 'Bingqing Wei'
import keras

from models.base import *

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
