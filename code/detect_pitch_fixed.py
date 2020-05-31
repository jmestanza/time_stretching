import librosa 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import find_peaks
from utils import separate_regions

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

    regions = separate_regions(is_frame_speech)

    for reg in regions:
        start,end = reg
        i = 0
        while (start+(i+1)*L + K <= end): # con esto no se sale de la region, no quiero casos borde
            x1 = audio[start+i*L:start+(i+1)*L]
            x2 = audio[start+i*L:start+(i+1)*L + K]
            indexes.append(start+i*L)
            split_audio.append([x1,x2])
            i+=1

    f0s_in_samples = []
    for i, x1_and_x2 in enumerate(split_audio):
        x1,x2 = x1_and_x2

        if is_frame_speech[indexes[i]]:
            full_autocorr = process_window_autocorrelation(x1, x2, K,L,w_type)
            autocorrelation = full_autocorr[len(full_autocorr)//2:]
            # puede que esto valga 0 en algun momento, cuidado
            f0_hat_in_samples = get_f0_from_autocorr(autocorrelation, show_demo)#np.argmax(autocorrelation)
            if show_demo:
                plot_autocorrelation_and_signal(x1,x2,full_autocorr,autocorrelation, figsize_=figsize)
         
        else:# si no es sonoro, lo computo como 0
            f0_hat_in_samples = 0
        f0s_in_samples.append(f0_hat_in_samples)
        if show_demo:
            show_demo = False
    return indexes, f0s_in_samples




def get_f0_from_autocorr(x,show_demo, h= 0, dist=16, prom = None, width = None, thresh = None):
    
    peaks, info = find_peaks(x, height=h, distance = dist, prominence= prom, width = None, threshold= thresh)

    if show_demo:
        plt.plot(x)
        plt.plot(peaks, x[peaks], "x")
        plt.plot(np.zeros_like(x), "--", color="gray")
        
    heights = info['peak_heights']
    arg_max1 = np.argmax(heights) # obtengo el mas grande
    first = peaks[arg_max1] # lo guardo

    heights = np.delete(heights,arg_max1) # borro de los demas picos
    peaks = np.delete(peaks,arg_max1) # tmabien su indice

    arg_max2 = np.argmax(heights) # busco el proximo mas grande

    second = peaks[arg_max2]
    #print("primer pico grande en: ",first)
    #print("segundo pico grande en: ",second)
    # tecnicamente siempre seocnd deberia ser menor en altura que first
    # if(second > first):
    #     print("paso algo raro")
    return abs(first-second)

#print(heights)