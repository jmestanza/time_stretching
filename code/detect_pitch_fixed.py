import librosa 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

def plot_autocorrelation_and_signal(x1,x2,fullacorr,autocorr, figsize_=(10, 12)):
    plt.figure(figsize=figsize_)

    plt.subplot(411)
    plt.plot(x1)
    plt.legend(["x1 ventana de L"])
    
    plt.subplot(412)
    plt.plot(x2)
    plt.legend(["x2 ventana de K+L"])
    
    plt.subplot(413)
    plt.plot(autocorr)
    plt.legend(["autocorrelacion"])
    
    plt.subplot(414)
    plt.plot(fullacorr)
    plt.legend(["full autocorrelacion"])
    
    plt.suptitle('autocorrelation testing')
    plt.show()
    return 


def process_window_autocorrelation(x1,x2,K,L,window_type):
    if window_type == "hanning":
        w1 = np.hanning(L) # da muy bien con hanning!!
        w2 = np.hanning(L+K)
    elif window_type == "rectangular":
        w1 = np.ones(L)
        w2 = np.ones(L+K) 

    return np.convolve(x1*w1,x2*w2)

def get_fundamental_frequency(audio, is_frame_speech,K,L,fs, w_type = "hanning", show_demo = False, figsize=(10, 12)):
    audio = audio[:len(is_frame_speech)]
    indexes = []
    split_audio = []
    for i in range(0,len(audio),L):
        if(i+L+K< len(audio)):
            x1 = audio[i:i+L]
            x2 = audio[i:i+L+K]
            indexes.append(i)
            split_audio.append([x1,x2])


    f0s_in_samples = []
    for i, x1_and_x2 in enumerate(split_audio):
        x1,x2 = x1_and_x2
        full_autocorr = process_window_autocorrelation(x1, x2, K,L,w_type)

        autocorrelation = full_autocorr[len(full_autocorr)//2:]

        if show_demo:
            plot_autocorrelation_and_signal(x1,x2,full_autocorr,autocorrelation, figsize_=figsize)
            show_demo = False
        if is_frame_speech[indexes[i]]:
            # puede que esto valga 0 en algun momento, cuidado
            f0_hat_in_samples = np.argmax(autocorrelation) 
        else:# si no es sonoro, lo computo como 0
            f0_hat_in_samples = 0
        f0s_in_samples.append(f0_hat_in_samples)

    return indexes, f0s_in_samples
