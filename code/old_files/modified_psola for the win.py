import librosa 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from math import ceil

class pitch_centered_segment:
    def __init__(self, P , t , samples_windowed):
        self.pitch = P
        self.t = t
        self.samples_windowed = samples_windowed

def plot_peaks(x,peaks):
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
    plt.show()

def search_pitch_samples(peak,indexes):
    a = 0
    b = len(indexes) - 1
    # bin search de cuando posiblemente NO esta en el arreglo

    while(a<b):
        middle = a + (b-a)//2
        if peak < indexes[middle]:
            b = middle - 1
        elif peak > indexes[middle]: # si peak esta en middle>=peak
            a = middle + 1
        else:
            return middle

    return b
            

def modified_psola(x, min_dist, indexes,f0_in_samples, speed, is_frame_speech, frame_len):

    x = x[:len(is_frame_speech)]
    total_time = len(x)
    total_new_time = ceil(total_time*(1/speed))
    new_audio = np.zeros(total_new_time)

    # en distance deberia poner una distancia minima en relacion a la frecuencia maxima
    # quiza podria venir solo con lo sonoro a peaks.
    # fmax = 500
    # fs/muestras = f0 => muestras = fs/f0 = 8000 / 500 = 16 muestras
    
    # la que estaba antes era: min_dist = pitch_samples-int(0.16*pitch_samples)
    
    peaks, _ = find_peaks(x, height=0, distance = min_dist)
    # side_len y voy saltando de a L, tengo la info de una ventana de L

    #plot_peaks(x,peaks) # debugging

    pitch_centered = []
    p_samples = int(np.mean(f0_in_samples))
    for i,peak in enumerate(peaks):
        if is_frame_speech[peak]:
            #p_samples =f0_in_samples[search_pitch_samples(peak, indexes)]
            if peak-p_samples>=0 and peak+p_samples <= len(x) - 1: 
                samples_windowed = x[peak-p_samples:peak+p_samples+1]*np.hamming(2*p_samples+1)
                pitch_centered.append(pitch_centered_segment(p_samples, peak, samples_windowed))
   
    
    num, den = speed.as_integer_ratio() # 3 / 2 = > num 3 , den 2 ;  4 1 

    #aca hago cosas con pitch
    for i, centered in enumerate(pitch_centered):
        t_ = ceil(centered.t * (1/speed))
        if t_-centered.pitch>=0 and t_+centered.pitch <= len(new_audio) - 1: 
            if speed>=1:
                if den > i%num: #  
                    new_audio[t_-centered.pitch:t_+centered.pitch+1] += centered.samples_windowed

    #aca con no pitch
    w_no_sonora = np.hamming(2*frame_len+1)
    for i in range(0,len(new_audio)):
        if i*frame_len < len(new_audio):
            if not is_frame_speech[i*frame_len]:
                if speed >=1:
                    if den > i%num:
                        t = i*frame_len+frame_len//2
                        t_ = ceil(t* (1/speed))
                        if t_-frame_len >0 and t_+frame_len < len(new_audio):
                            new_audio[t_-frame_len:t_+frame_len+1] += x[t-frame_len:t+frame_len+1]*w_no_sonora

    return new_audio