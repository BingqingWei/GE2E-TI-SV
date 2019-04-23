__author__ = 'Bingqing Wei'
import numpy as np
import math
import random
import gc
from config import *
from data import wav2spectro
from norm import *

infer_enroll_path = os.path.join(config.infer_path, 'enroll')
infer_verif_path = os.path.join(config.infer_path, 'verif')

class BatchGenerator:
    def __init__(self):
        self.buffers = None
        self.reset()

    def reset(self):
        self.buffers = []
        for dataset in config.dataset:
            self.buffers.append(Buffer(dataset=dataset))

    def gen_batch(self, sels, frames):
        batches = []
        r_sels = []
        for buffer, sel, frame in zip(self.buffers, sels, frames):
            new_batch, new_sel = self.random_batch(buffer, selected_files=sel, frames=frame)
            batches.append(new_batch)
            r_sels.append(new_sel)
        return np.concatenate(batches, axis=1), r_sels

    def gen_batch2(self):
        batches = []
        if config.mode == 'train':
            frames = np.random.randint(config.min_frames, config.max_frames)
        else: frames = config.mid_frames
        for buffer in self.buffers:
            batch1, batch2 = self.random_batch2(buffer, frames=frames)
            batches.append(np.concatenate([batch1, batch2], axis=1))

        # shape=(frames, N * M * 2 * len(datasets), mels)
        return np.concatenate(batches, axis=1)


    def random_batch(self, buffer, selected_files=None, frames=None):
        return buffer.sample(sel_speakers=selected_files, frames=frames)

    def random_batch2(self, buffer, frames=None):
        return buffer.sample2(frames=frames)

class ValidBatchGenerator(BatchGenerator):
    def __init__(self, nb_batches=5):
        self.nb_batches = nb_batches
        super(ValidBatchGenerator, self).__init__()

    def reset(self):
        self.buffers = []
        for dataset in config.dataset:
            self.buffers.append(Buffer(dataset=dataset, K_N=self.nb_batches,
                                       recycle=True, mode='test'))

class Buffer:
    def __init__(self, dataset='voxceleb',
                 K_N=config.K_N, recycle=False, mode=config.mode):
        """
        :param dataset: vctk or voxceleb
        :param K_N: K_N * N speakers to be sampled
        :param K_M: K_M * M wav files per speaker to be loaded
        :param recycle: if True, no flushing will be performed
        """

        self.dataset = dataset
        self.counter = 0
        self.K_N = K_N * config.N
        self.K_M = 22 # minimum utters wrt npy file
        self.recycle = recycle
        self.count_down = self.calcCountDown()
        if mode == 'train':
            self.data_path = os.path.join(config.train_path, dataset)
        else:
            self.data_path = os.path.join(config.test_path, dataset)

        self.buffer = None
        self.flush()
        if config.debug:
            if self.recycle:
                print('recyling contents, count_down infinite')
            else:
                print('buffer countdown: ', self.count_down)

    def calcCountDown(self):
        return int(config.K_N ** 2 * self.K_M / config.M)

    def update(self, npy_list):
        """
        :param npy_list:
        :return: whether to flush the buffer
        """
        self.K_N = min(self.K_N, len(npy_list))
        self.count_down = self.calcCountDown()
        self.counter = 0
        return self.K_N != len(npy_list) or self.buffer is None or self.recycle

    def flush(self):
        npy_list = os.listdir(self.data_path)
        do_flush = self.update(npy_list)
        if not do_flush: return

        self.buffer = []
        sel_speakers = random.sample(npy_list, self.K_N)
        for file in sel_speakers:
            utters = np.load(os.path.join(self.data_path, file))
            utter_index = random.sample(range(utters.shape[0]), self.K_M)
            self.buffer.append(utters[utter_index])
        self.buffer = np.concatenate(self.buffer, axis=0)

    def sample2(self, speaker_num=config.N, utter_num=config.M, frames=None):
        sel_speakers = random.sample(range(self.K_N), speaker_num)
        batch_1, batch_2 = [], []
        for i in sel_speakers:
            utters = self.buffer[i * self.K_M:(i + 1) * self.K_M, :]
            utter_index = random.sample(range(utters.shape[0]), 2 * utter_num)
            batch_1.append(utters[utter_index[:utter_num]])
            batch_2.append(utters[utter_index[utter_num:]])
        batch_1 = np.concatenate(batch_1, axis=0)
        batch_2 = np.concatenate(batch_2, axis=0)
        if frames is None:
            if config.mode == 'train':
                frames = np.random.randint(config.min_frames, config.max_frames)
            else: frames = config.mid_frames
        batch_1 = batch_1[:, :, :frames]
        batch_2 = batch_2[:, :, :frames]

        # shape = (frames, N * M, 40)
        batch_1 = np.transpose(batch_1, axes=(2, 0, 1))
        batch_2 = np.transpose(batch_2, axes=(2, 0, 1))
        self.counter += 2
        if self.counter >= self.count_down:
            self.flush()

        return normalize_batch(batch_1, self.dataset), normalize_batch(batch_2, self.dataset)

    def sample(self, speaker_num=config.N, utter_num=config.M,
               sel_speakers=None, frames=None):
        if sel_speakers is None:
            sel_speakers = random.sample(range(self.K_N), speaker_num)

        batch = []
        for i in sel_speakers:
            utters = self.buffer[i * self.K_M:(i + 1) * self.K_M, :]
            utter_index = np.random.randint(0, utters.shape[0], utter_num)
            batch.append(utters[utter_index])
        batch = np.concatenate(batch, axis=0)
        if config.mode == 'train':
            if frames is None:
                frames = np.random.randint(config.min_frames, config.max_frames)
            batch = batch[:, :, :frames]
        else:
            if frames is None:
                frames = int((config.min_frames + config.max_frames) / 2)
            batch = batch[:, :, :frames]

        # shape = (frames, N * M, 40)
        batch = np.transpose(batch, axes=(2, 0, 1))
        self.counter += 1
        if self.counter >= self.count_down:
            self.flush()

        return normalize_batch(batch, self.dataset), sel_speakers

def gen_infer_batches():
    """
    :return: enrolls, verifs
    """
    def print_error_and_exit(process):
        msg = '{} utterances are too bad to process.\n' \
              'The reasons can be:\n' \
              '1. too short.\n' \
              '2. too much silence.\n' \
              '3. not loud enough.'.format(process)
        print(msg)
        raise Exception()

    enroll_utters = []
    verif_utters = []
    for file in os.listdir(infer_enroll_path):
        enroll_utters.extend(wav2spectro(os.path.join(infer_enroll_path, file)))
    for file in os.listdir(infer_verif_path):
        verif_utters.extend(wav2spectro(os.path.join(infer_verif_path, file)))

    if len(enroll_utters) < 2:
        print_error_and_exit('Enrollment')
    if len(verif_utters) < 1:
        print_error_and_exit('Verification')

    enroll_utters = np.transpose(np.array(enroll_utters), axes=(2, 0, 1))
    verif_utters = np.transpose(np.array(verif_utters), axes=(2, 0, 1))
    return normalize_batch(enroll_utters, config.dataset[0]), normalize_batch(verif_utters, config.dataset[0])

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    #buffer = Buffer(dataset='voxceleb')
    gen = ValidBatchGenerator()
    for i in range(100):
        gen.gen_batch2()
