__author__ = 'Bingqing Wei'
import os

data_path = r'E:\Data'
work_dir = r'.\data'

class Config: pass

"""
during execution:
modifying anything other than the config.mode is not recommended
"""

config_dict = {
    # Data
    'noise_path': os.path.join(work_dir, 'noise'),          # noise dataset directory
    'train_path': os.path.join(work_dir, 'train_tisv'),     # train dataset directory
    'test_path': os.path.join(work_dir, 'test_tisv'),       # test dataset directory
    'model_path': os.path.join(work_dir, 'tisv_model'),     # save paths
    'infer_path': os.path.join(work_dir, 'infer'),

    # Preprocessing
    'nb_noises': 64,                                        # how many of the noise files to choose
    'max_ckpts': 6,                                         # max checkpoints to keep
    'sr': 8000,                                             # sample rate
    'nfft': 512,                                            # fft kernel size
    'window': 0.025,                                        # window length (ms)
    'hop': 0.01,                                            # hop size (ms)
    'max_frames': 180,                                      # number of max frames
    'min_frames': 140,                                      # number of min frames
    'mels':40,

    # Model
    'nb_hidden': 256,                                       # number of hidden units
    'nb_proj': 128,                                         # number of projection units
    'nb_layers': 3,                                         # number of LSTM_Projection layers
    'loss':'softmax',
    'flush_thres': 200,                                     # flushing threshold of the buffer
    'K_N': 20,                                              # K_N * N spearkers will be stored in buffer
    'K_M': 4,                                               # each speaker in buffer will have 4 wave files
    'gpu_fraction': 0.8,                                    # gpu fraction

    # Session
    'mode': 'train',                                        # train or test
    'N': 5,                                                 # number of speakers per batch
    'M': 4,                                                 # number of utterances per speaker
    'lr': 0.01,
    'optim': ['sgd',                                        # type of the optimizer
              {'beta1': 0.9, 'beta2': 0.999}],              # additional parameters
    'decay': 'cosine',
    'nb_iters': 1e5,                                        # max iterations
    'save_per_iters': 3000,                                 # save models per X iterations
    'decay_per_iters': 6000,                               # decay learning rate per X iterations
    'log_per_iters': 100,                                   # log info per X iterations
    'summary_per_iters':50,                                 # write summary per X iterations
    'verbose': True,
    'dataset': ['voxceleb'],                                # datasets to be used. if mode is set to infer,
                                                            # then all datasets are ignored
                                                            # in test mode, only one dataset is allowed
    'weights': [1.0],                                       # weights for each dataset

    # Debug
    'debug': True,                                          # turn on debug info output
    'redirect_stdout': False,
}

assert config_dict['mode'] in ['train', 'test', 'infer']
assert len(config_dict['dataset']) != 0
if config_dict['mode'] == 'test':
    assert len(config_dict['dataset']) == 1

for dataset in config_dict['dataset']:
    assert dataset in ['voxceleb', 'vctk']
assert len(config_dict['weights']) == len(config_dict['dataset'])

config_dict['mid_frames'] = int((config_dict['max_frames'] + config_dict['min_frames']) / 2)


config = Config()
config.__dict__.update(config_dict)