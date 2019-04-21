# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 11:04:28 2019

@author: glebv
"""

import time
import pyaudio
import wave
import os
from adv import infer_enroll_path, infer_verif_path
from config import *
from main import *

def clear_and_make(fdir):
    if os.path.exists(fdir):
        shutil.rmtree(fdir)
    os.mkdir(fdir)

class Recorder(object):
    def __init__(self, channels=1, rate=44100, frames_per_buffer=1024):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer

    def open(self, fname, mode='wb'):
        return RecordingFile(fname, mode, self.channels, self.rate,
                            self.frames_per_buffer)

class RecordingFile(object):
    def __init__(self, fname, mode, channels, 
                rate, frames_per_buffer):
        self.fname = fname
        self.mode = mode
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self._pa = pyaudio.PyAudio()
        self.wavefile = self._prepare_file(self.fname, self.mode)
        self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        self.close()

    def record(self, duration):
       
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.frames_per_buffer)
        for _ in range(int(self.rate / self.frames_per_buffer * duration)):
            audio = self._stream.read(self.frames_per_buffer)
            self.wavefile.writeframes(audio)
        return None

    def start_recording(self):

        self._stream = self._pa.open(format=pyaudio.paInt16,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.frames_per_buffer,
                                        stream_callback=self.get_callback())
        self._stream.start_stream()
        return self

    def stop_recording(self):
        self._stream.stop_stream()
        return self

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            self.wavefile.writeframes(in_data)
            return in_data, pyaudio.paContinue
        return callback


    def close(self):
        self._stream.close()
        self._pa.terminate()
        self.wavefile.close()

    def _prepare_file(self, fname, mode='wb'):
        wavefile = wave.open(fname, mode)
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.rate)
        return wavefile

if __name__ == '__main__':
    recorder = Recorder()
    clear_and_make(infer_verif_path)
    clear_and_make(infer_enroll_path)

    print('Enrollment start')
    for i in range(4):
        wav_file = recorder.open(os.path.join(infer_enroll_path, 'record-{}.wav'.format(i)))
        input('press any key to record')
        wav_file.start_recording()
        input('press any key to stop recording')
        wav_file.close()

    print('Verification start')
    for i in range(4):
        wav_file = recorder.open(os.path.join(infer_verif_path, 'record-{}.wav'.format(i)))
        input('press any key to record')
        wav_file.start_recording()
        input('press any key to stop recording')
        wav_file.close()

    config.norm = True
    config.redirect_stdout = False
    config.mode = 'infer'
    main()


