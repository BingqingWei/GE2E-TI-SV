__author__ = 'Bingqing Wei'
import numpy as np
import math
import random
import gc
from config import *
from data import wav2spectro

infer_enroll_path = os.path.join(config.infer_path, 'enroll')
infer_verif_path = os.path.join(config.infer_path, 'verif')

class Buffer:
    def __init__(self, K_N=(10 * config.N), K_M=(2 * config.M), flush_thres=1.5):
        """
        :param K_N:
        :param K_M:
        :param flush_thres: should be greater than 1
        """
        if not config.mode == 'train': flush_thres = 0.2

        self.flush_thres = flush_thres
        self.count_down = int(math.sqrt(K_N * K_M * flush_thres))
        self.counter = 0
        self.K_N = K_N
        self.K_M = K_M
        if config.mode == 'train':
            self.data_path = os.path.join(config.train_path, config.dataset)
        else:
            self.data_path = os.path.join(config.train_path, config.dataset)
        self.buffer = None
        self.flush()

    def update(self, npy_list):
        """
        :param npy_list:
        :return: whether to flush the buffer
        """
        self.K_N = min(self.K_N, len(npy_list))
        self.count_down = int(math.sqrt(self.K_N * self.K_M * self.flush_thres))
        self.counter = 0
        return self.K_N != len(npy_list) or self.buffer is None

    def flush(self):
        npy_list = os.listdir(self.data_path)
        do_flush = self.update(npy_list)
        if not do_flush: return

        if config.debug: print('flushing buffer')

        del self.buffer
        gc.collect()
        self.buffer = []

        sel_speakers = random.sample(npy_list, self.K_N)
        for file in sel_speakers:
            utters = np.load(os.path.join(self.data_path, file))
            utter_index = np.random.randint(0, utters.shape[0], self.K_M)
            self.buffer.append(utters[utter_index])

        self.buffer = np.concatenate(self.buffer, axis=0)

    def sample(self, speaker_num=config.N, utter_num=config.M, sel_speakers=None, frames=None):
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

        return batch, sel_speakers

buffer = Buffer()

def reset_buffer():
    global buffer
    buffer = Buffer()

def random_batch(speaker_num=config.N, utter_num=config.M, selected_files=None, frames=None):
    return buffer.sample(speaker_num, utter_num, sel_speakers=selected_files, frames=frames)

def gen_infer_batches():
    """
    :return: enrolls, verifs
    """

    enroll_utters = []
    verif_utters = []
    for file in os.listdir(infer_enroll_path):
        enroll_utters.extend(wav2spectro(os.path.join(infer_enroll_path, file)))
    for file in os.listdir(infer_verif_path):
        verif_utters.extend(wav2spectro(os.path.join(infer_verif_path, file)))

    enroll_utters = np.transpose(np.array(enroll_utters), axes=(2, 0, 1))
    verif_utters = np.transpose(np.array(verif_utters), axes=(2, 0, 1))
    return enroll_utters, verif_utters
