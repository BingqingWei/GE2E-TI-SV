import tensorflow as tf
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import random
import os
import glob
import shutil
from config import *

def clear_and_make(fdir):
    if os.path.exists(fdir):
        shutil.rmtree(fdir)
    os.mkdir(fdir)

def get_latest_ckpt(fpath):
    files = glob.glob(os.path.join(fpath, 'model.ckpt-*.meta'))
    files = [x[:-5] for x in files]
    files.sort()
    if len(files) == 0: return None
    return files[-1]

def normalize(x):
    return x / tf.sqrt(tf.reduce_sum(x ** 2, axis=-1, keepdims=True) + 1e-6)

def embedd2center(embedd):
    return normalize(tf.reduce_mean(tf.reshape(embedd, shape=[config.N, config.M, -1]), axis=1))

def cossim(x, y, normalized=True):
    if normalized:
        return tf.reduce_sum(x * y)
    else:
        x_norm = tf.sqrt(tf.reduce_sum(x ** 2) + 1e-6)
        y_norm = tf.sqrt(tf.reduce_sum(y ** 2) + 1e-6)
        return tf.reduce_sum(x * y) / x_norm / y_norm

def center_similarity(embedd_1, embedd_2):
    center_1 = embedd2center(embedd_1)
    center_2 = embedd2center(embedd_2)
    return tf.matmul(center_1, tf.transpose(center_2, perm=[1, 0]))

def similarity(embedded, w, b, N=config.N, M=config.M, center=None):
    embedded_split = tf.reshape(embedded, shape=[N, M, -1])

    if center is None:
        center = normalize(tf.reduce_mean(embedded_split, axis=1))
        center_except = normalize(tf.reshape(tf.reduce_sum(embedded_split, axis=1, keepdims=True) - embedded_split,
                                             shape=[N * M, -1]) / (M - 1))

        S = tf.concat(
            [tf.concat([tf.reduce_sum(center_except[i * M:(i + 1) * M, :] * embedded_split[j, :, :], axis=1,
                                      keepdims=True) if i == j
                        else tf.reduce_sum(center[i:(i + 1), :] * embedded_split[j, :, :], axis=1, keepdims=True) for i
                        in range(N)],
                       axis=1) for j in range(N)], axis=0)
    else:
        # center[i, :] * embedded_slit[j, :, :] is element-wise multiplication
        # therefore it needs to use reduce_sum to get vectors dot product
        S = tf.concat([
            tf.concat([tf.reduce_sum(center[i, :] * embedded_split[j, :, :], axis=1,
                                     keepdims=True)
                       for i in range(N)], axis=1) for j in range(N)], axis=0)

    # shape = (N * M, N)
    S = tf.abs(w) * S + b
    return S


def loss_cal(S, name='softmax', N=config.N, M=config.M):
    # S_{j i, j}
    S_correct = tf.concat([S[i * M:(i + 1) * M, i:(i + 1)] for i in range(N)], axis=0)  # colored entries in Fig.1

    if name == 'softmax':
        total = -tf.reduce_sum(S_correct - tf.log(tf.reduce_sum(tf.exp(S), axis=1, keepdims=True) + 1e-6))
    elif name == 'contrast':
        S_sig = tf.sigmoid(S)
        S_sig = tf.concat([tf.concat([0 * S_sig[i * M:(i + 1) * M, j:(j + 1)] if i == j
                                      else S_sig[i * M:(i + 1) * M, j:(j + 1)] for j in range(N)], axis=1)
                           for i in range(N)], axis=0)
        total = tf.reduce_sum(1 - tf.sigmoid(S_correct) + tf.reduce_max(S_sig, axis=1, keepdims=True))
    else:
        raise AssertionError('loss type should be softmax or contrast!')
    return total

def optim(lr):
    assert config.optim[0] in ['sgd', 'rmsprop', 'adam']
    if config.optim[0] == 'sgd':
        return tf.train.GradientDescentOptimizer(lr)
    elif config.optim[0] == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr, **config.optim[1])
    else:
        return tf.train.AdamOptimizer(lr, **config.optim[1])

def applyDecay(start_ler, global_step):
    assert config.decay in ['cosine', 'exp']
    if config.decay == 'cosin':
        return tf.train.cosine_decay(start_ler, global_step, config.decay_per_iters)
    else:
        return tf.train.exponential_decay(start_ler, global_step, config.decay_per_iters,
                                          decay_rate=0.5, staircase=True)
