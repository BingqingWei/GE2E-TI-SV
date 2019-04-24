# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 11:04:28 2019

@author: glebv
"""

import time
import pyaudio
import wave
import os
from main import *
from config import *
from utils import clear_and_make

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

def record_one(recorder, fdir, fname, min_second=4):
    while True:
        input('press enter to record')
        wav_file = recorder.open(os.path.join(fdir, fname))
        start = time.time()
        wav_file.start_recording()
        input('press enter to stop recording')
        end = time.time()
        wav_file.close()
        seconds = end - start
        if seconds < min_second:
            print('Too short, pleas try again')
        else:
            break
    print('saved to {}\n'.format(os.path.join(fdir, fname)))


if __name__ == '__main__':
    assert config.mode == 'infer'
    assert not config.redirect_stdout

    recorder = Recorder()
    clear_and_make(infer_verif_path)
    clear_and_make(infer_enroll_path)

    print('Enrollment start')
    for i in range(4):
        record_one(recorder, infer_enroll_path, 'record-{}.wav'.format(i), min_second=3)
    print('Verification start')
    for i in range(2):
        record_one(recorder, infer_verif_path, 'record-{}.wav'.format(i), min_second=5)
    main()
