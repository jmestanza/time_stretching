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
            

def find_max_probability_f0(start,end,indexes,sample_f0):
    f0s_in_reg = []
    for i,idx in enumerate(indexes):
        if start<= idx and idx <= end:
        #   if sample_f0[idx] != 0:
            f0s_in_reg.append(sample_f0[i])
    hist, bin_edges = np.histogram(np.array(f0s_in_reg), bins=150) 
    plt.plot(hist)    
    plt.show()
    print("retorno: ",bin_edges[np.argmax(hist)])
    return int(bin_edges[np.argmax(hist)])

def modified_psola(x, min_dist, indexes,f0_in_samples, speed, is_frame_speech, frame_len, regions):

    x = x[:len(is_frame_speech)]
    total_time = len(x)
    total_new_time = ceil(total_time*(1/speed))
    new_audio = np.zeros(total_new_time)

    num, den = speed.as_integer_ratio() # 3 / 2 = > num 3 , den 2 ;  4 1 

    for reg in regions:
        start,end = reg
        if is_frame_speech[start]:
            print("Procesando la parte sonora...")
            p_samples = find_max_probability_f0(start,end,indexes,f0_in_samples)
            print(p_samples)
            peaks, _ = find_peaks(x, height=0, distance = p_samples - p_samples*0.16)
            w_sonora = np.hamming(2*p_samples+1)
            pitch_centered = []
            for i,peak in enumerate(peaks):
                if peak <= start and peak <= end:
                    if peak-p_samples>=0 and peak+p_samples <= len(x) - 1: 
                        samples_windowed = x[peak-p_samples:peak+p_samples+1]*w_sonora
                        pitch_centered.append(pitch_centered_segment(p_samples, peak, samples_windowed))

            for i, centered in enumerate(pitch_centered):
                t_ = ceil(centered.t * (1/speed))
                if t_-centered.pitch>=0 and t_+centered.pitch <= len(new_audio) - 1: 
                    if speed>=1:
                        if den > i%num: #  
                            print("aniadiendo a new_audio")
                            new_audio[t_-centered.pitch:t_+centered.pitch+1] += centered.samples_windowed

            print("Termine de procesar la parte sonora...")
    
        else:# si es no sonora
            w_no_sonora = np.hamming(2*p_samples+1)
            for i in range(end-start):
                t = start + i*p_samples + p_samples//2
                t_ = ceil(t*(1/speed))
                if speed >=1:
                    if den > i%num:
                        if t-p_samples>=0 and t_+p_samples < len(new_audio):
                            new_audio[t_-p_samples:t_+p_samples+1] += x[t-p_samples:t+p_samples+1]*w_no_sonora

    return new_audio