import tensorflow as tf
import numpy as np
import os
import math
from utils import *
from config import *

class Model:
    def __init__(self):
        if config.mode == 'train':
            self.batch = tf.placeholder(shape=[None, config.N * config.M, config.mels], dtype=tf.float32)
            w = tf.get_variable('w', initializer=np.array([10], dtype=np.float32))
            b = tf.get_variable('b', initializer=np.array([-5], dtype=np.float32))
            self.lr = tf.placeholder(dtype=tf.float32)
            global_step = tf.Variable(0, name='global_step', trainable=False)

            embedded = self.build_model(self.batch)
            s_mat = similarity(embedded, w, b)

            if config.verbose:
                print('embedded size: ', embedded.shape)
                print('similarity matrix size: ', s_mat.shape)
            self.loss = loss_cal(s_mat, name=config.loss)

            trainable_vars = tf.trainable_variables()
            optimizer = optim(self.lr)

            grads, params = zip(*optimizer.compute_gradients(self.loss))
            grads_clip, _ = tf.clip_by_global_norm(grads, 3.0)

            # 0.01 gradient scale for w and b, 0.5 gradient scale for projection nodes
            grads_rescale = [0.01 * g for g in grads_clip[:2]]
            for g, p in zip(grads_clip[2:], params[2:]):
                if 'projection' in p.name:
                    grads_rescale.append(0.5 * g)
                else:
                    grads_rescale.append(g)

            self.train_op = optimizer.apply_gradients(zip(grads_rescale, params), global_step=global_step)

            variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
            if config.verbose: print('total variables:', variable_count)

            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()

        elif config.mode == 'test':
            self.batch = tf.placeholder(shape=[None, config.N * config.M * 2, config.mels], dtype=tf.float32)
            embedded = self.build_model(self.batch)
            # concatenate [enroll, verif]
            enroll_embed = tf.reduce_mean(
                tf.reshape(embedded[:config.N * config.M, :], shape=[config.N, config.M, -1]), axis=1)
            verif_embed = embedded[config.N * config.M:, :]

            self.s_mat = similarity(embedded=verif_embed, w=1.0, b=0.0, center=enroll_embed)

        else:
            self.enroll = tf.placeholder(shape=[None, None, config.mels], dtype=tf.float32)
            self.verif = tf.placeholder(shape=[None, None, config.mels], dtype=tf.float32)
            enroll_size = tf.shape(self.enroll)[1]
            batch = tf.concat([self.enroll, self.verif], axis=1)
            embedded = self.build_model(batch)
            enroll_embed = normalize(embedded[:enroll_size, :])
            verif_embed = normalize(embedded[enroll_size:, :])

            enroll_center = tf.reduce_mean(enroll_embed, axis=0)
            verif_center = tf.reduce_mean(verif_embed, axis=0)
            self.s = tf.reduce_sum(enroll_center * verif_center, axis=0)

        self.saver = tf.train.Saver()

    def build_model(self, batch):
        with tf.variable_scope('lstm'):
            cells = [tf.contrib.rnn.LSTMCell(num_units=config.nb_hidden, num_proj=config.nb_proj)
                     for i in range(config.nb_layers)]
            lstm = tf.contrib.rnn.MultiRNNCell(cells)
            outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)
            embedded = outputs[-1]

            # shape = (N * M, nb_proj)
            embedded = normalize(embedded)
        return embedded


    def train(self, sess, path):
        assert config.mode == 'train'
        sess.run(tf.global_variables_initializer())

        model_path = os.path.join(path, 'check_point')
        log_path = os.path.join(path, 'logs')

        os.makedirs(model_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        writer = tf.summary.FileWriter(log_path, sess.graph)
        lr_factor = 1
        loss_acc = 0
        for i in range(int(config.nb_iters)):
            batch, _ = random_batch()
            _, loss_cur, summary = sess.run([self.train_op, self.loss, self.merged],
                                            feed_dict={self.batch: batch,
                                                       self.lr: config.lr * lr_factor})
            loss_acc += loss_cur

            if i % 10 == 0:
                writer.add_summary(summary, i)

            if (i + 1) % 100 == 0:
                if config.verbose: print('(iter : %d) loss: %.4f' % ((i + 1), loss_acc / 100))
                loss_acc = 0

            if (i + 1) % 10000 == 0:
                lr_factor /= 2
                if config.verbose: print('learning rate is decayed! current lr : ', config.lr * lr_factor)

            if (i + 1) % 3000 == 0:
                self.saver.save(sess, os.path.join(path, 'check_point', 'model.ckpt'), global_step=i // 3000)
                if config.verbose: print('model is saved!')

    def test(self, sess, path, nb_batch_thres=5, nb_batch_test=100):
        assert config.mode == 'test'
        def cal_ff(s, thres):
            s_thres = s > thres

            far = sum([np.sum(s_thres[i]) - np.sum(s_thres[i, :, i]) for i in range(config.N)]) / \
                  (config.N - 1) / config.M / config.N
            frr = sum([config.M - np.sum(s_thres[i][:, i]) for i in range(config.N)]) / config.M / config.N
            return far, frr

        def gen_batch():
            enroll_batch, selected_files = random_batch(frames=160)
            verif_batch, _ = random_batch(selected_files=selected_files, frames=160)
            return np.concatenate([enroll_batch, verif_batch], axis=1)

        self.saver.restore(sess, path)

        config.train = True
        reset_buffer()
        s_mats = []
        for i in range(nb_batch_thres):
            s = sess.run(self.s_mat, feed_dict={self.batch: gen_batch()})
            s = s.reshape([config.N, config.M, -1])
            s_mats.append(s)

        diff = math.inf
        EER = 0
        THRES = 0

        for thres in [0.01 * i + 0.5 for i in range(50)]:
            fars = []
            frrs = []
            for s in s_mats:
                far, frr = cal_ff(s, thres)
                fars.append(far)
                frrs.append(frr)

            FAR = np.mean(fars)
            FRR = np.mean(frrs)
            if diff > abs(FAR - FRR):
                diff = abs(FAR - FRR)
                THRES = thres
                EER = (FAR + FRR) / 2.0
        print('(validation) thres: {}, EER: {}'.format(THRES, EER))

        config.train = False
        reset_buffer()
        EERS = []
        for i in range(nb_batch_test):
            s = sess.run(self.s_mat, feed_dict={self.batch: gen_batch()})
            s = s.reshape([config.N, config.M, -1])

            far, frr = cal_ff(s, THRES)
            EERS.append((far + frr) / 2)

        EER = np.mean(EERS)
        print('(test) EER: {}'.format(EER))


    def infer(self, sess, path, thres=0.57):
        self.saver.restore(sess, path)
        enrolls, verifs = gen_infer_batches()
        s = sess.run(self.s, feed_dict={
            self.enroll: enrolls,
            self.verif: verifs
        })
        if s > thres:
            print('same speaker')
        else:
            print('different speakers')
