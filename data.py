from tqdm import tqdm

from utils import *
from config import *

vctk_path = os.path.join(data_path, 'VCTK')
ds_path = os.path.join(data_path, 'DS_10283_1942')
voxceleb_path = os.path.join(data_path, 'voxceleb1')

clean_path = os.path.join(ds_path, 'clean_testset_wav')
noisy_path = os.path.join(ds_path, 'noisy_testset_wav')

avg_frames = int((config.max_frames + config.min_frames) / 2.0)
hop_frames = int(avg_frames / 2)
utter_min_len = (config.max_frames * config.hop + config.window) * config.sr

def extract_noise():
    print('starting noise extraction')
    os.makedirs(config.noise_path, exist_ok=True)
    batch_frames = config.N * config.M * config.frames

    stacked_noise = []
    stacked_len = 0
    k = 0
    for i, path in enumerate(tqdm(os.listdir(clean_path))):

        clean, sr = librosa.core.load(os.path.join(clean_path, path), sr=8000)
        noisy, _ = librosa.core.load(os.path.join(noisy_path, path), sr=sr)
        noise = clean - noisy
        S = librosa.core.stft(y=noise, n_fft=config.nfft, win_length=int(config.window * sr),
                              hop_length=int(config.hop * sr))

        stacked_noise.append(S)
        stacked_len += S.shape[1]
        if stacked_len < batch_frames: continue

        stacked_noise = np.concatenate(stacked_noise, axis=1)[:, :batch_frames]
        np.save(os.path.join(config.noise_path, 'noise_{}.npy'.format(k)), stacked_noise)
        stacked_noise = []
        stacked_len = 0
        k += 1

    print('noise extraction ended, total {} files'.format(k))

def wav2spectro(utter_path):
    utterances_spec = []
    utter, sr = librosa.core.load(utter_path, config.sr)
    intervals = librosa.effects.split(utter, top_db=20)
    for interval in intervals:
        if (interval[1] - interval[0]) > utter_min_len:
            utter_part = utter[interval[0]:interval[1]]
            S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
                                  win_length=int(config.window * sr), hop_length=int(config.hop * sr))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=config.mels)
            S = np.log10(np.dot(mel_basis, S) + 1e-6)

            if config.mode != 'infer':
                '''
                NOTE: each interval in utterance only extracts 2 samples
                '''
                utterances_spec.append(S[:, :config.max_frames])
                utterances_spec.append(S[:, -config.max_frames:])
            else:
                max_steps = int((S.shape[1] - avg_frames) / hop_frames) + 1
                for i in range(max_steps):
                    utterances_spec.append(S[:, hop_frames * i : hop_frames * i + avg_frames])
    return utterances_spec

def save_spectrogram(speakers, train_path, test_path, test_split, start_sid=0):
    """
    :param speakers: list[list[file_path]]
    :param train_path:
    :param test_path:
    :param test_split:
    :param start_sid: resume processing starting from
    :return:
    """
    assert 0 <= test_split < 1
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    total_speaker_num = len(speakers)

    train_speaker_num = int(total_speaker_num * (1 - test_split))

    print('total speaker number : {}'.format(len(speakers)))
    nb_utters = np.array([len(x) for x in speakers])
    print('min nb_utterances: {}, max_nb_utterances: {}'.format(np.min(nb_utters), np.max(nb_utters)))
    print('train : {}, test : {}'.format(train_speaker_num, total_speaker_num - train_speaker_num))

    for i, files in enumerate(tqdm(speakers[start_sid:])):
        utterances_spec = []
        for utter_path in files:
            utterances_spec.extend(wav2spectro(utter_path))

        if i + start_sid < train_speaker_num:
            np.save(os.path.join(train_path, 'speaker_{}.npy'.format(i + start_sid)), utterances_spec)
        else:
            np.save(os.path.join(test_path, 'speaker_{}.npy'.format(i + start_sid - train_speaker_num)), utterances_spec)

def save_spectrogram_voxceleb(test_split=0.1, start_sid=0):
    print('processing voxceleb dataset')
    audio_path = os.path.join(voxceleb_path, 'wav')

    train_path = os.path.join(config.train_path, 'voxceleb')
    test_path = os.path.join(config.test_path, 'voxceleb')

    speakers = []
    for folder in os.listdir(audio_path):
        speaker_path = os.path.join(audio_path, folder)
        utters = []
        for sub_folder in os.listdir(speaker_path):
            sub_utter_path = os.path.join(speaker_path, sub_folder)
            for wav_fname in os.listdir(sub_utter_path):
                utters.append(os.path.join(sub_utter_path, wav_fname))
        speakers.append(utters)

    save_spectrogram(speakers, train_path, test_path, test_split, start_sid)


def save_spectrogram_vctk(test_split=0.1, start_sid=0):
    print('processing vctk dataset')
    audio_path = os.path.join(vctk_path, 'data')

    train_path = os.path.join(config.train_path, 'vctk')
    test_path = os.path.join(config.test_path, 'vctk')

    speakers = []
    for folder in os.listdir(audio_path):
        speaker_path = os.path.join(audio_path, folder)
        speakers.append(list(os.listdir(speaker_path)))

    save_spectrogram(speakers, train_path, test_path, test_split, start_sid)

if __name__ == '__main__':
    #save_spectrogram_vctk()
    #extract_noise()
    save_spectrogram_voxceleb(start_sid=1088)
