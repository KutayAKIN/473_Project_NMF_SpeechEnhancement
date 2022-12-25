import os

import numpy as np
import matplotlib.pyplot as plt

from functions.utilized_functions.util import *
from functions.nmf_algorithm.NMF import train_NMF, test_NMF 


def WienerMask(V_estimate, B, W, K_speech):
    B_speech = B[:,:K_speech]
    W_speech = W[:K_speech,:]
    tilde_S = np.matmul(B_speech, W_speech)
    
    B_noise = B[:,K_speech:]
    W_noise = W[K_speech:,:]
    tilde_N = np.matmul(B_noise, W_noise)
    
    M_wiener = np.power(tilde_S, 2) / (np.power(tilde_S, 2) + np.power(tilde_N, 2))
    
    S_hat = M_wiener * V_estimate
    N_hat = (1 - M_wiener) * V_estimate
    
    return S_hat, N_hat

def plot_mag_spec(mixture_mag_spec, clean_mag_spec, enhanced_mag_spec, args):
    figure, axis = plt.subplots(3, 1, sharex = False)
    
    axis[0].matshow(mag2dB(mixture_mag_spec), origin = 'lower', aspect = 'auto', cmap ='jet')
    axis[1].matshow(mag2dB(clean_mag_spec), origin='lower', aspect='auto', cmap='jet')
    axis[2].matshow(mag2dB(enhanced_mag_spec), origin='lower', aspect='auto', cmap='jet')
    
    axis[0].set_title('SNR = ' + str(args.SNR) +' db, mixture magnitude spectrogram in dB')
    axis[1].set_title("Original test speech audio magnitude spectrogram in dB")
    axis[2].set_title("Enhanced test speech audio magnitude spectrogram in dB")
    
    for j1 in range(3):
        axis[j1].xaxis.tick_bottom()
    plt.tight_layout()
    figure = plt.gcf()
    plt.show()
    
    return figure

def save_plot(figure, magnitude_spectogram_path, test_clean_speech_name):
    makedirs(magnitude_spectogram_path)
    nmf_result_path = os.path.join(magnitude_spectogram_path,"result_for_%s.jpeg" %(os.path.splitext(test_clean_speech_name)[0]))
    figure.savefig(nmf_result_path, dpi = 300)
    

def main(DIRECTORIES, args):
    
    OUTPUT_PATH = DIRECTORIES['OUTPUT_PATH']
    TEST_SPEECH_PATH = DIRECTORIES['TEST_SPEECH_PATH']
    
    #----------------TRAINING STEP----------------------
    print("NMF Algorithm: Start Trainning Step, beta = %s" %(args.beta))
    
    TRAIN_SOURCE_PATH = os.path.join(OUTPUT_PATH, 'train_source')
    TRAIN_SPEECH_SOURCE_PATH = os.path.join(TRAIN_SOURCE_PATH, "train_speech_merged_audio.wav")
    TRAIN_NOISE_SOURCE_PATH = os.path.join(TRAIN_SOURCE_PATH, "train_noise_merged_audio.wav") 
    
    # Obtain mag specs 
    hop_size = int(0.25*args.window_size) # For hamming windows, overlap usually chosen as %75 of fft window size, hence hop size is %25 of fft window size.
    V_s = abs(audio2spec(TRAIN_SPEECH_SOURCE_PATH, args.fs, args.fft, hop_size))
    V_n = abs(audio2spec(TRAIN_NOISE_SOURCE_PATH, args.fs, args.fft, hop_size))
    
    # NMF approximation for V ~= BW 
    K_total = args.K_speech + args.K_noise 
    V = np.concatenate((V_s, V_n), axis=1)
    B_train, W_train = train_NMF(V, args.max_train_i, args.epsilon, K_total, args.beta)
    
    print("NMF Algoritm: End Trainning Step")
    
    
    #-------------------DECOMPOSTION STEP -------------------
    print("NMF Algoritm: Start Decomposition Step, beta = %s" %(args.beta))
    
    enhanced_audio_path = os.path.join(OUTPUT_PATH, 'enhanced_audio', 'beta_' + str(args.beta))
    makedirs(enhanced_audio_path)
    
    TEST_SOURCE_PATH = os.path.join(OUTPUT_PATH, 'test_source')
    test_source_names = [na for na in os.listdir(TEST_SOURCE_PATH) if na.lower().endswith(".wav")]
    test_clean_speech_names = [na for na in os.listdir(TEST_SPEECH_PATH) if na.lower().endswith(".wav")]
    bb = 0
    for kk in range(len(test_source_names)):

        test_source_name = test_source_names[kk]
        test_source_sample_path = os.path.join(TEST_SOURCE_PATH, test_source_name)
        
        test_source_mag_spec= audio2spec(test_source_sample_path, args.fs, args.fft, hop_size) # spectrogram for noisy
        
        V_test_source = abs(test_source_mag_spec) # magnitude spectrogram for noisy 
        W_test_source, _ = test_NMF(V_test_source, B_train, args.max_test_i, args.epsilon, K_total, args.beta) # new encoding vector obtained through nmf algorithm
        
        speech_after_mask, noise_after_mask = WienerMask(V_test_source, B_train, W_test_source, args.K_speech) # magnitude spectrogram for enhanced 
        enhanced_audio = spec2audio(speech_after_mask, np.angle(test_source_mag_spec), hop_size) # reconstruct audio 
        
        # write enhanced audio
        enhanced_audio_sample_path = os.path.join(enhanced_audio_path, "enhanced_%s" %test_source_name)
        write_audio(enhanced_audio_sample_path, enhanced_audio, args.fs)
        

        
        
        #------------------- Plotting the magnitude sectograms ------------------      
        if args.bool_plot:
            
            test_clean_speech_name = test_clean_speech_names[bb]
            bb = bb + 1
            test_clean_speech_sample_path = os.path.join(TEST_SPEECH_PATH, test_clean_speech_name)
            
            test_clean_speech_mag_spec = audio2spec(test_clean_speech_sample_path, 16000, args.fft, hop_size)
            
            # Plot magnitude spectogram 
            figure = plot_mag_spec(test_source_mag_spec, test_clean_speech_mag_spec, speech_after_mask, args)
            
            #------- Save the results ------------
            if args.bool_plot_save:
                magnitude_spectogram_path = os.path.join(OUTPUT_PATH, "magnitude_spectogram_plots")
                save_plot(figure, magnitude_spectogram_path, test_clean_speech_name)
                     
    print("NMF Algoritm: End Decomposition Step")
 


if __name__ == '__main__':
    main()


