__author__ = 'Bingqing Wei'
import tensorflow as tf
import numpy as np
from utils import similarity, embedd2center, loss_cal
import time

if __name__ == '__main__':
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.Session(config=tf_config)

    N, M = 16, 7
    embed = tf.placeholder(dtype=tf.float32, shape=(N * 2 * M, 128))

    # new loss
    embed_1 = embed[:N * M]
    embed_2 = embed[N * M:]
    center_1 = embedd2center(embed_1, N, M)
    center_2 = embedd2center(embed_2, N, M)
    new_loss = loss_cal(similarity(embed_1, 1.0, 0.0, N, M, center_2), name='softmax', N=N, M=M) + \
               loss_cal(similarity(embed_2, 1.0, 0.0, N, M, center_1), name='softmax', N=N, M=M)

    # oldloss
    old_loss = loss_cal(similarity(embed, 1.0, 0.0, N, M * 2), N=N, M=M * 2)
    sess.run(tf.global_variables_initializer())

    arr = np.random.rand(N * M * 2, 128)

    times = []
    print('Calculating old loss')
    for i in range(1000):
        start = time.time()
        _ = sess.run(old_loss, feed_dict={embed: arr})
        end = time.time()
        times.append(end - start)
    print('Used: {} seconds'.format(np.mean(times)))

    times = []
    print('Calculating new loss')
    for i in range(1000):
        start = time.time()
        _ = sess.run(new_loss, feed_dict={embed: arr})
        end = time.time()
        times.append(end - start)
    print('Used: {} seconds'.format(np.mean(times)))
