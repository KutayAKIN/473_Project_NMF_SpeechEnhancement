
import os
import numpy as np

from functions.utilized_functions.util import *

def add_noise(speech, noise, SNR):
    if  len(noise) > len(speech):
        noise = noise[0:len(speech)]
    else:
        speech = speech[0:len(noise)]
        
    noise = adjustPower(speech, noise, SNR)    
    mixture = speech + noise
    return mixture

    
def main(DIRECTORIES, args):
    
    print("Started making test audio.")
    
    TEST_SPEECH_PATH = DIRECTORIES['TEST_SPEECH_PATH']
    TEST_NOISE_PATH = DIRECTORIES['TEST_NOISE_PATH']
    OUTPUT_PATH = DIRECTORIES['OUTPUT_PATH']
    
    # set output path
    TEST_SOURCE_PATH = os.path.join(OUTPUT_PATH, 'test_source')
    DIRECTORIES['TEST_SOURCE_PATH'] = TEST_SOURCE_PATH
    makedirs(TEST_SOURCE_PATH)
    
    test_speech_names = [na for na in os.listdir(TEST_SPEECH_PATH) if na.lower().endswith(".wav")]
    test_noise_names = [na for na in os.listdir(TEST_NOISE_PATH) if na.lower().endswith(".wav")]
    
    for test_speech_name in test_speech_names:
        test_speech_sample_path = os.path.join(TEST_SPEECH_PATH, test_speech_name)
        test_speech_audio = read_audio(test_speech_sample_path, target_fs = 16000)
        
        for test_noise_name in test_noise_names:
            test_noise_sample_path = os.path.join(TEST_NOISE_PATH, test_noise_name)
            test_noise_audio = read_audio(test_noise_sample_path, target_fs = args.fs)

            test_mixture_audio = add_noise(speech = test_speech_audio, noise = test_noise_audio, SNR = 0)
            test_mixture_name = os.path.join("%s_%s.wav" % (os.path.splitext(test_speech_name)[0], os.path.splitext(test_noise_name)[0]))
            
            test_mixture_name_path = os.path.join(TEST_SOURCE_PATH, test_mixture_name)
            write_audio(test_mixture_name_path, test_mixture_audio, args.fs)
    
    print("Finished making test audio.")
    
    

if __name__ == '__main__':
    main()