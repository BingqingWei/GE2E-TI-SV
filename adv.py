__author__ = 'Bingqing Wei'
import numpy as np
import math
import random
import gc
from config import *
from data import wav2spectro

infer_enroll_path = os.path.join(config.infer_path, 'enroll')
infer_verif_path = os.path.join(config.infer_path, 'verif')
regularize = lambda x: np.expand_dims(np.array(x), axis=0)

vox_mels_mean = [-3.02734108, -2.4091782, -2.09676263, -2.14122718, -2.27263581,
                 -2.29585195, -2.32880616, -2.37786825, -2.46338034, -2.61611401,
                 -2.77438724, -2.93828297, -3.10202187, -3.23742286, -3.36306108,
                 -3.48222725, -3.57529019, -3.65443268, -3.71971935, -3.7786455,
                 -3.81745664, -3.83876491, -3.85377475, -3.85456816, -3.87392065,
                 -3.90093468, -3.94232578, -3.99431193, -4.04679665, -4.09095003,
                 -4.11386166, -4.12873658, -4.15824979, -4.23098748, -4.32646621,
                 -4.41495825, -4.49644065, -4.55911425, -4.61389892, -4.92468198]
vox_mels_mean = regularize(vox_mels_mean)

vox_mels_std = [0.99771783, 1.04186374, 1.10912619, 1.14970144, 1.17117287,
                1.21564558, 1.25839678, 1.28691782, 1.31305374, 1.33677685,
                1.35104074, 1.34753574, 1.32738919, 1.3035531, 1.2803645,
                1.26185408, 1.24903902, 1.23968701, 1.23258394, 1.22656812,
                1.22164563, 1.21799986, 1.21723812, 1.21799928, 1.21524414,
                1.20804512, 1.19542939, 1.17863671, 1.15915348, 1.1411941,
                1.13073703, 1.12894445, 1.12746134, 1.11255488, 1.0874017,
                1.06459398, 1.04497637, 1.0271965, 1.01555041, 0.92520637]
vox_mels_std = regularize(vox_mels_std)

vctk_mels_mean = [-2.20264042, -2.12578949, -1.95736143, -2.06582957, -2.41532483,
       -2.50703519, -2.47427188, -2.51433437, -2.62260055, -2.84876494,
       -3.1197115 , -3.36351928, -3.57581458, -3.72864567, -3.81024453,
       -3.84810831, -3.8422651 , -3.83540179, -3.84128367, -3.87757678,
       -3.91938242, -3.94998426, -3.97376936, -3.97255475, -3.97646164,
       -3.98795116, -4.01634964, -4.05846623, -4.11231269, -4.18206786,
       -4.22262974, -4.23214157, -4.23298933, -4.24964223, -4.26909383,
       -4.29360511, -4.32171169, -4.35202394, -4.41442044, -4.7973996 ]
vctk_mels_mean = regularize(vctk_mels_mean)

vctk_mels_std = [0.6176385 , 1.0793721 , 1.66721085, 1.81359393, 1.68252913,
       1.66599848, 1.72437261, 1.73588632, 1.72109329, 1.68165345,
       1.6251986 , 1.56902801, 1.50822147, 1.44305163, 1.39288623,
       1.36586753, 1.35992709, 1.35837829, 1.35898632, 1.3554012 ,
       1.34880004, 1.33844371, 1.3263748 , 1.31874043, 1.30917467,
       1.29611295, 1.27763259, 1.25410011, 1.22551194, 1.19106073,
       1.16733687, 1.16243238, 1.16666534, 1.16238329, 1.1533377 ,
       1.14389358, 1.13477486, 1.12475346, 1.10918917, 0.97668915]
vctk_mels_std = regularize(vctk_mels_std)


def normalize_batch(batch, dataset):
    if dataset == 'voxceleb':
        batch -= vox_mels_mean
        batch /= vox_mels_std
    else:
        batch -= vctk_mels_mean
        batch /= vctk_mels_std
    return batch

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

class Buffer:
    def __init__(self, flush_thres=config.flush_thres, dataset='voxceleb'):
        """
        :param K_N:
        :param K_M:
        :param flush_thres: should be greater than 1
        """
        if config.mode != 'train': flush_thres = 0.4
        self.dataset = dataset

        self.flush_thres = flush_thres
        self.count_down = int(config.K_N * config.K_M * flush_thres)
        self.counter = 0
        self.K_N = config.K_N * config.N
        self.K_M = config.K_M * config.M
        if config.mode == 'train':
            self.data_path = os.path.join(config.train_path, dataset)
        else:
            self.data_path = os.path.join(config.train_path, dataset)
        self.buffer = None
        self.flush()
        if config.debug: print('buffer countdown: ', self.count_down)

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

        del self.buffer
        gc.collect()
        self.buffer = []

        sel_speakers = random.sample(npy_list, self.K_N)
        for file in sel_speakers:
            utters = np.load(os.path.join(self.data_path, file))
            utter_index = np.random.randint(0, utters.shape[0], self.K_M)
            self.buffer.append(utters[utter_index])

        self.buffer = np.concatenate(self.buffer, axis=0)

    def sample2(self, speaker_num=config.N, utter_num=config.M, frames=None):
        sel_speakers = random.sample(range(self.K_N), speaker_num)
        batch_1, batch_2 = [], []
        for i in sel_speakers:
            utters = self.buffer[i * self.K_M:(i + 1) * self.K_M, :]
            utter_index = np.random.randint(0, utters.shape[0], utter_num)
            batch_1.append(utters[utter_index])
            utter_index = np.random.randint(0, utters.shape[0], utter_num)
            batch_2.append(utters[utter_index])
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

    enroll_utters = []
    verif_utters = []
    for file in os.listdir(infer_enroll_path):
        enroll_utters.extend(wav2spectro(os.path.join(infer_enroll_path, file)))
    for file in os.listdir(infer_verif_path):
        verif_utters.extend(wav2spectro(os.path.join(infer_verif_path, file)))

    enroll_utters = np.transpose(np.array(enroll_utters), axes=(2, 0, 1))
    verif_utters = np.transpose(np.array(verif_utters), axes=(2, 0, 1))
    return enroll_utters, verif_utters

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    buffer = Buffer(dataset='voxceleb')

