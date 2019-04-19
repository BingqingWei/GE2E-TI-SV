__author__ = 'Bingqing Wei'

import tensorflow as tf
from models.base import *

def residual_block(prev, filters, kernel):
    x = tf.nn.relu(prev)
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel, padding='same', use_bias=False)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel, padding='same', use_bias=False)(x)
    x = tf.nn.relu(x)
    return tf.add(x, prev)


class ResidualCNN(Model):
    def build_model(self, batch):
        # shape = (batch_size, time_teps, mels)
        batch = tf.transpose(batch, perm=[1, 0, 2])
        x = tf.keras.layers.BatchNormalization()(batch)
        x = tf.keras.layers.Conv1D(filters=32, kernel_size=(16,), padding='same', activation='relu')(x)
        for i in range(16):
            x = residual_block(x, filters=32, kernel=(32,))
        encoded = tf.keras.layers.Conv1D(filters=1, kernel_size=(32,), activation='relu')(x)
        encoded = tf.squeeze(encoded, axis=-1)
        return encoded



