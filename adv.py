__author__ = 'Bingqing Wei'
import random
from data import wav2spectro
from norm import *
from config import *

import math

infer_enroll_path = os.path.join(config.infer_path, 'enroll')
infer_verif_path = os.path.join(config.infer_path, 'verif')

class BatchGenerator:
    def __init__(self, n_batch):
        self.buffer = None
        self.reset()
        self.n_batch = n_batch

    def reset(self):
        self.buffer = Buffer(dataset=config.dataset)

    def gen_batch(self):
        if config.mode == 'train':
            frames = np.random.randint(config.min_frames, config.max_frames)
        else: frames = config.mid_frames
        batches = self.buffer.sampleN(frames, N=self.n_batch)
        # shape=(frames, N * M * n_batch, mels)
        return np.concatenate(batches, axis=1)

class ValidBatchGenerator(BatchGenerator):
    def __init__(self, K_N=1):
        self.reset_K_N = K_N
        super(ValidBatchGenerator, self).__init__(n_batch=config.n_batch)

    def reset(self):
        self.buffer = Buffer(dataset=config.dataset, K_N=self.reset_K_N, recycle=True, mode='test')

class Buffer:
    def __init__(self, dataset='voxceleb', K_N=config.K_N, recycle=False, mode=config.mode):
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

    def sampleN(self, frames, speaker_num=config.N, utter_num=config.M, N=2):
        sel_speakers = random.sample(range(self.K_N), speaker_num)
        batches = [[] for _ in range(N)]
        for i in sel_speakers:
            utters = self.buffer[i * self.K_M:(i + 1) * self.K_M, :]
            utter_index = random.sample(range(utters.shape[0]), N * utter_num)
            for j in range(N):
                batches[j].append(utters[utter_index[utter_num * j : utter_num * (j + 1)]])
        batches = [np.concatenate(x) for x in batches]
        batches = [normalize_batch(np.transpose(x[:, :, :frames], axes=(2, 0, 1)), self.dataset) for x in batches]
        self.counter += 2
        if self.counter >= self.count_down:
            self.flush()
        return batches

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
    pass
