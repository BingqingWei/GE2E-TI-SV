import tensorflow as tf
import numpy as np
import os
import math
import sys
from utils import *
from config import *
from adv import *

class Model:
    def __init__(self):
        if config.mode == 'infer':
            self.enroll = tf.placeholder(shape=[None, None, config.mels], dtype=tf.float32)
            self.verif = tf.placeholder(shape=[None, None, config.mels], dtype=tf.float32)
            enroll_size = tf.shape(self.enroll)[1]
            batch = tf.concat([self.enroll, self.verif], axis=1)
            embedded = self.build_model(batch)
            enroll_embed = normalize(embedded[:enroll_size, :])
            verif_embed = normalize(embedded[enroll_size:, :])

            enroll_center = normalize(tf.reduce_mean(enroll_embed, axis=0))
            verif_center = normalize(tf.reduce_mean(verif_embed, axis=0))
            self.s = tf.reduce_sum(enroll_center * verif_center, axis=0)
        else:
            self.batch = tf.placeholder(shape=[None, config.N * config.M * 2 * len(config.dataset), config.mels], dtype=tf.float32)
            embedded = self.build_model(self.batch)
            offset_base = config.N * config.M * 2
            embedd_01 = [embedded[i * offset_base + config.N * config.M: (i + 1) * offset_base, :]
                         for i in range(len(config.dataset))]
            embedd_02 = [embedded[i * offset_base :i * offset_base + config.N * config.M, :]
                         for i in range(len(config.dataset))]

            if config.mode == 'train':
                w = tf.get_variable('w', initializer=np.array([10], dtype=np.float32))
                b = tf.get_variable('b', initializer=np.array([-5], dtype=np.float32))
                center_01 = [embedd2center(embedd) for embedd in embedd_01]
                center_02 = [embedd2center(embedd) for embedd in embedd_02]

                if config.verbose: print('embedded size: ', embedd_01[0].shape)

                s_mat_01 = [similarity(embedded=embedd, w=w, b=b, center=center)
                            for embedd, center in zip(embedd_01, center_02)]
                s_mat_02 = [similarity(embedded=embedd, w=w, b=b, center=center)
                            for embedd, center in zip(embedd_02, center_01)]
                if config.verbose: print('similarity matrix size: ', s_mat_01[0].shape)

                self.loss = 0.0
                for s_1, s_2, weight in zip(s_mat_01, s_mat_02, config.weights):
                    self.loss += weight * (loss_cal(s_1, name=config.loss) + loss_cal(s_2, name=config.loss))

                trainable_vars = tf.trainable_variables()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.lr = applyDecay(config.lr, self.global_step)
                optimizer = optim(self.lr)

                grads, params = zip(*optimizer.compute_gradients(self.loss))

                # 0.01 gradient scale for w and b, 0.5 gradient scale for projection nodes
                grads_rescale = [0.01 * g for g in grads[:2]]
                for g, p in zip(grads[2:], params[2:]):
                    if 'projection' in p.name:
                        grads_rescale.append(0.5 * g)
                    else:
                        grads_rescale.append(g)

                if config.debug: tf.summary.scalar('gradient_norm', tf.global_norm(grads_rescale))
                grads_rescale, _ = tf.clip_by_global_norm(grads_rescale, 3.0)
                self.train_op = optimizer.apply_gradients(zip(grads_rescale, params), global_step=self.global_step)
                variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
                if config.verbose: print('total variables:', variable_count)
                tf.summary.scalar('loss', self.loss)
                self.merged = tf.summary.merge_all()

            elif config.mode == 'test':
                self.s_mat = center_similarity(embedd_01, embedd_02)
            else: raise ValueError()
        self.saver = tf.train.Saver()

    def build_model(self, batch):
        """
        :param batch: (time_steps, batch_size, mels)
        :return:
        """
        raise NotImplementedError()


    def train(self, sess, path):
        assert config.mode == 'train'
        sess.run(tf.global_variables_initializer())
        generator = BatchGenerator()
        valid_generator = ValidBatchGenerator(nb_batches=config.nb_valid * 2)

        model_path = os.path.join(path, 'check_point')
        log_path = os.path.join(path, 'logs')
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        writer = tf.summary.FileWriter(log_path, sess.graph)
        loss_acc = 0
        best_valid = np.inf
        for i in range(int(config.nb_iters)):
            _, loss_cur, summary = sess.run([self.train_op, self.loss, self.merged],
                                            feed_dict={self.batch: generator.gen_batch2()})
            loss_acc += loss_cur

            if (i + 1) % config.log_per_iters == 0:
                if config.verbose: print('(iter : %d) loss: %.4f' % ((i + 1), loss_acc / config.log_per_iters))
                loss_acc = 0
                if config.redirect_stdout: sys.stdout.flush()

            if (i + 1) % config.summary_per_iters == 0:
                writer.add_summary(summary, i)
                writer.flush()

            if (i + 1) % config.save_per_iters == 0:
                valid_loss = self.valid(sess, valid_generator)
                if valid_loss > best_valid:
                    print('validation loss is too large, skipping')
                    continue
                best_valid = min(best_valid, valid_loss)
                self.saver.save(sess, os.path.join(path, 'check_point', 'model.ckpt'),
                                global_step=i // config.save_per_iters)
                if config.verbose: print('model is saved')


    def valid(self, sess, generator):
        loss_acc = 0
        for i in range(config.nb_valid):
            _, loss_cur = sess.run([self.train_op, self.loss],
                                   feed_dict={self.batch: generator.gen_batch2()})
            loss_acc += loss_cur
        print('validation loss: {}'.format(loss_acc / config.nb_valid))
        return loss_acc / config.nb_valid


    def test(self, sess, path, nb_batch_thres=100, nb_batch_test=1000):
        assert config.mode == 'test'
        def cal_ff(s, thres):
            s_thres = s > thres
            far = (np.sum(s_thres) - config.N) / float(config.N * (config.N - 1))
            frr = (config.N - np.trace(s_thres)) / float(config.N)
            return far, frr

        self.saver.restore(sess, path)
        config.train = True
        generator = BatchGenerator()
        s_mats = []
        for i in range(nb_batch_thres):
            s = sess.run(self.s_mat, feed_dict={self.batch: generator.gen_batch2()})
            s_mats.append(s)

        diff, EER, THRES = math.inf, 0, 0
        for thres in [0.01 * i for i in range(100)]:
            fars, frrs = [], []
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
        generator.reset()
        EERS = []
        for i in range(nb_batch_test):
            s = sess.run(self.s_mat, feed_dict={self.batch: generator.gen_batch2()})
            far, frr = cal_ff(s, THRES)
            EERS.append((far + frr) / 2)

        EER = np.mean(EERS)
        print('(test) EER: {}'.format(EER))


    def infer(self, sess, path, thres=0.41):
        assert config.mode == 'infer'
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
