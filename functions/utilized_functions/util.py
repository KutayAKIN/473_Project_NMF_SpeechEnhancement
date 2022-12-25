import os
import librosa
import soundfile
import sys

import numpy as np

def makedirs(path):
    if not os.path.exists(path):
        print("Make directories : {}".format(path))
        os.makedirs(path)

def read_audio(path, target_fs =  None):
    (audio_wav, fs) = soundfile.read(path)
    if audio_wav.ndim > 1:
        audio_wav = np.mean(audio_wav, axis = 1)
    if target_fs is not None and fs != target_fs:
        audio_wav = librosa.resample(audio_wav, orig_sr = fs, target_sr = target_fs)
    return audio_wav

def write_audio(audio_path, data, fs):
    soundfile.write(file = audio_path, data = data, samplerate = fs)

def audio2spec(audio_path, fs, fft_size, hop_size):
    audio_wav = read_audio(audio_path, fs)
    audio_spec = librosa.stft(audio_wav, n_fft = fft_size, hop_length = hop_size, window = 'hamming')  
    audio_spec[np.where(audio_spec == 0)[0]] = sys.float_info.epsilon
    return audio_spec
    
def spec2audio(magnitude, phase, hop_size):
    complx = magnitude * np.exp(1j*phase)
    audio = librosa.istft(complx, hop_length = hop_size, window = 'hamming')
    return audio

def adjustPower(speech, noise, SNR):
    target = pow(speech)/(pow(noise) * dB2pow(SNR)) * noise
    return target

def pow(input):
    pow = np.sqrt(np.sum(input**2))
    return pow

def rms(input):
    rms = np.sqrt(np.mean(input**2))
    return rms

def dB2pow(SNR_dB):
    SNR_pow = np.power(10,(SNR_dB/10))
    return SNR_pow

def mag2dB(input):
    output = 20*np.log10(np.abs(input)**2)
    return output

def get_merged_audio(path_read_folder, audio_type, args):
    
    merged_audio = []   
    if audio_type=='speech': 
        for ii in list(range(1,10)) + list(range(11, 20)):
            if ii == 8:
                continue
            else:
                # wav, rate = read_wav(filename , sr=args.fs)
                wav  = path_read_folder.audios[ii].numpy()
                rate = args.fs
                merged_audio.append(np.reshape(wav, (wav.shape[0], ))/rms(wav))
    
    elif audio_type=='noise':
        for filename in glob.glob(os.path.join(path_read_folder, '*.wav')):
            wav, rate = read_audio(filename, target_fs=args.fs)
            end = int (args.fs * 30) 
            merged_audio.append(wav[0:end]) # just using first 25 seconds data
        
    merged_audio=np.hstack(merged_audio)
    return merged_audio,rate