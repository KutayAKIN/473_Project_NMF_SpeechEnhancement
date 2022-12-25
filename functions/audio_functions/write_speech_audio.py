"""
Created on Tue Dec 20 13:23:23 2022

@author: EYYPT
"""
import os
import numpy as np

from functions.utilized_functions.util import *

def write_speech(SPEECH_PATH, ds, num_sample, type):
    samples = []
    if type == 'test':
        for ii in range(int(num_sample/9)):
            samples += list(range(10*ii, 10*ii + 1))
    elif type == 'train':
        for ii in range(int(num_sample/9)):
            samples += list(range(10*ii + 1, 10*(ii + 1)))
    
    for kk in samples:
        fs = 16000
        speech_audio  = ds.audios[kk].numpy()
        speech_audio = np.reshape(speech_audio, (speech_audio.shape[0], ))/rms(speech_audio)
        speech_name = os.path.join("%s_%s.wav" %(type, kk))
        speech_audio_path = os.path.join(SPEECH_PATH, speech_name)
        write_audio(speech_audio_path, speech_audio, fs)

def main(DIRECTORIES, args, ds_train):    
    # Writing the speech train data. 
    print("Write the speech audio for train data.")
    
    TRAIN_SPEECH_PATH = DIRECTORIES['TRAIN_SPEECH_PATH']
    write_speech(TRAIN_SPEECH_PATH, ds_train, args.speech_sample, 'train')
    
    print("Finished writing train audio.")
    
    # Writing the speech test data. 
    print("Write the speech audio for test data.")
    
    TEST_SPEECH_PATH = DIRECTORIES['TEST_SPEECH_PATH']
    write_speech(TEST_SPEECH_PATH, ds_train, args.speech_sample, 'test')
    
    print("Finished writing test audio.")
    

    

if __name__ == '__main__':
    main()