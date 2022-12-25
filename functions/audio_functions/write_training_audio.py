
import os
import glob
import numpy as np

from functions.utilized_functions.util import *

def get_train_audio(TRAIN_PATH, type, fs):
    
    train_names = [na for na in os.listdir(TRAIN_PATH) if na.lower().endswith(".wav")]
    
    audio_merged = []  
    if type == 'speech':
        for train_speech_name in train_names:
            train_speech_sample_path = os.path.join(TRAIN_PATH, train_speech_name)

            train_speech_audio = read_audio(train_speech_sample_path, target_fs = 16000)
            audio_merged.append(train_speech_audio)

    elif type=='noise':
        for train_noise_name in train_names:
            train_speech_sample_path = os.path.join(TRAIN_PATH, train_noise_name)

            train_noise_audio = read_audio(train_speech_sample_path, target_fs = fs)
            end = int(fs * 20) # just using first 20 seconds data
            audio_merged.append(train_noise_audio[0:end]) 
        
    audio_merged = np.hstack(audio_merged)
    return audio_merged
    
def main(DIRECTORIES, args):
    
    print("Started making training audio.")
    
    TRAIN_SPEECH_PATH = DIRECTORIES['TRAIN_SPEECH_PATH']
    TRAIN_NOISE_PATH = DIRECTORIES['TRAIN_NOISE_PATH']
    OUTPUT_PATH = DIRECTORIES['OUTPUT_PATH']
    
    # Make directory for the output path
    TRAIN_SOURCE_PATH = os.path.join(OUTPUT_PATH, 'train_source')
    DIRECTORIES['TRAIN_SOURCE_PATH'] = TRAIN_SOURCE_PATH
    
    makedirs(TRAIN_SOURCE_PATH)
    
    # Merged the train speech/noise samples
    merged_speech_audio = get_train_audio(TRAIN_SPEECH_PATH, 'speech', 16000)
    merged_noise_audio = get_train_audio(TRAIN_NOISE_PATH, 'noise', args.fs)
   
    # Adjust power level of noise according to SNR
    merged_noise_audio = adjustPower(merged_speech_audio, merged_noise_audio, args.SNR)
    
    # write wav files
    write_audio(TRAIN_SOURCE_PATH+'/train_speech_merged_audio.wav', merged_speech_audio, 16000)
    write_audio(TRAIN_SOURCE_PATH+'/train_noise_merged_audio.wav', merged_noise_audio, args.fs)
    

    print("Finished making training audio.")
    
    

if __name__ == '__main__':
    main()
