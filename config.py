__author__ = 'Bingqing Wei'
import os

data_path = r'E:\Data'
work_dir = r'.\data-16000'

class Config: pass

"""
during execution:
modifying anything other than the config.mode is not recommended
"""

config_dict = {
    # Data
    'train_path': os.path.join(work_dir, 'train_tisv'),     # train dataset directory
    'test_path': os.path.join(work_dir, 'test_tisv'),       # test dataset directory
    'model_path': os.path.join(work_dir, 'tisv_model'),     # save paths
    'infer_path': os.path.join(work_dir, 'infer'),

    # Preprocessing
    'max_ckpts': 6,                                         # max checkpoints to keep
    'sr': 16000,                                            # sample rate
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
    'K_N': 10,                                              # K_N * N spearkers will be stored in buffer
    'gpu_fraction': 0.6,                                    # gpu fraction

    # Session
    'mode': 'infer',                                        # train or test
    'N': 16,                                                # number of speakers per batch
    'M': 7,                                                 # number of utterances per speaker
    'lr': 0.001,
    'optim': ['adam',                                       # type of the optimizer
              {'beta1': 0.9, 'beta2': 0.999}],              # additional parameters
    'decay': 'cosine',
    'nb_iters': 1e5,                                        # max iterations
    'save_per_iters': 3000,                                 # save models per X iterations
    'decay_per_iters': 8000,                                # decay learning rate per X iterations
    'log_per_iters': 100,                                   # log info per X iterations
    'summary_per_iters':100,                                # write summary per X iterations
    'dataset': ['voxceleb'],                                # datasets to be used. if mode is set to infer,
                                                            # then all datasets are ignored
                                                            # in test mode, only one dataset is allowed
    'weights': [1.0],                                       # weights for each dataset
    'nb_valid': 50,                                         # number of batches to be used in validation

    # Debug
    'verbose': True,
    'debug': True,                                          # turn on debug info output
    'redirect_stdout': False,
    'norm': False,                                          # if True, buffers will normalize the batches
    'redirect_fname': 'test-3000.txt'
}

assert config_dict['M'] * 2 <= 22
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
