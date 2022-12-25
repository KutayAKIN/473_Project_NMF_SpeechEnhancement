
import os
import argparse
import deeplake

import numpy as np

from functions.audio_functions import write_speech_audio, write_training_audio, write_test_audio
from functions import nmf_main


def get_args():
    parser = argparse.ArgumentParser(description="Nonnegative Matrix Factorization for speech enhancement")
    
    #-----------------Signal Preprocessing Parameters----------------- 
    parser.add_argument('--fs', default=16000, type = int) # Set the sampling frequency
    parser.add_argument('--fft', default=1024, type = int) # Set the FFT point size
    parser.add_argument('--window_size', default=256, type = int) # Window size for the FFT
    parser.add_argument('--SNR', default = 20, type = int) # SNR level 
    
    #----------------------Dataset Parameters-------------------------
    parser.add_argument('--speech_sample', default = 18, type = int) # The number of total speech samples, it used with 9:1 ratio for training and testin
    parser.add_argument('--bool_online_dataset_speech', default = False, type = bool) # Get the test and train data of speech from online dataset
    parser.add_argument('--bool_merged_dataset', default = False, type = bool) # Make train and test sets.

    #----------------- Non-negative Matrix Factorization parameters ----------------- 
    parser.add_argument('--max_train_i', default = 300, type = int) # Maximum number of iterations for training step
    parser.add_argument('--max_test_i',  default = 60, type = int) # Maximum number of iterations for decomposition step 
    parser.add_argument('--K_speech', default = 128, type = int) # Number of basis of speech signals, inner dimension of NMF. 
    parser.add_argument('--K_noise', default = 128, type = int) # Number of basis of noise signals, inner dimension of NMF. 
    parser.add_argument('--epsilon', default = 0.0005, type = float) # Convergence threshold 
    parser.add_argument('--beta', default = 2, type = int) # Beta value to determine type of cost function, beta = 0 IS-divergence, beta = 1 KL-divergence, beta = 2 Euclidean Distance

    #----------------- Plotting and Saving the Results ----------------- 
    parser.add_argument('--bool_plot', default = True, type = bool) # If user wants to plot the results
    parser.add_argument('--bool_plot_save', default = True, type = bool) # If user wants to save the plots.

    args = parser.parse_args() 
    return args

def get_dirs():
    
    CURRENT_PATH = os.getcwd() # current directory path 
    TRAIN_NOISE_PATH = os.path.join(CURRENT_PATH,"data/train/noise") #directory path containing noise files for training.(training/noise)
    TRAIN_SPEECH_PATH =  os.path.join(CURRENT_PATH,"data/train/speech") # directory path containing speech files for training. (training/speech)
    TEST_NOISE_PATH = os.path.join(CURRENT_PATH,"data/test/noise") # directory path containing noise files for test. (test/noise)
    TEST_SPEECH_PATH = os.path.join(CURRENT_PATH,"data/test/speech") # directory path containing speech files for test. (test/speech)
    OUTPUT_PATH = os.path.join(CURRENT_PATH,"output") # path for output directory

    DIRECTORIES = {}   
    DIRECTORIES['CURRENT_PATH'] = CURRENT_PATH
    DIRECTORIES['TRAIN_NOISE_PATH'] = TRAIN_NOISE_PATH
    DIRECTORIES['TRAIN_SPEECH_PATH'] = TRAIN_SPEECH_PATH
    DIRECTORIES['TEST_NOISE_PATH'] = TEST_NOISE_PATH
    DIRECTORIES['TEST_SPEECH_PATH'] = TEST_SPEECH_PATH 
    DIRECTORIES['OUTPUT_PATH'] = OUTPUT_PATH
    
    return DIRECTORIES

if __name__ == '__main__':
    
    
    args = get_args()
    DIRECTORIES = get_dirs()
    
    np.random.seed(0)
    
    # We pull the speech data from online TIMIT library, 
    # for noise we downloaded the FUSS library so we do 
    # not need to connect online for noise data.
    if (args.bool_online_dataset_speech):
        ds_train = deeplake.load("hub://activeloop/timit-train")
        write_speech_audio.main(DIRECTORIES, args, ds_train)

    if(args.bool_merged_dataset):
        write_test_audio.main(DIRECTORIES, args)
        write_training_audio.main(DIRECTORIES, args)
                 
        
    nmf_main.main(DIRECTORIES, args) # run NMF alogorithm for speech enhancement
    
