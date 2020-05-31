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
            

def find_max_probability_f0(reg_number,start,end,indexes,sample_f0):
    f0s_in_reg = []
    for i,idx in enumerate(indexes):
        if start<= idx and idx <= end:
        #   if sample_f0[idx] != 0:
            f0s_in_reg.append(sample_f0[i])
    hist, bin_edges = np.histogram(np.array(f0s_in_reg), bins=150) 
    #plt.plot(hist)    
    #plt.xlabel("Distribucion de f0_hat en samples")
    #plt.ylabel("Frecuencia")
    #plt.show()
    return int(bin_edges[np.argmax(hist)])

def modified_psola(x, indexes, f0_in_samples, percent,percent_pitch, speed, is_frame_speech, regions,pitch_version, pitch_dif_mode,unvoiced_samples=360):

    x = x[:len(is_frame_speech)]
    total_time = len(x)
    total_new_time = ceil(total_time*(1/speed))
    new_audio = np.zeros(total_new_time)

    num, den = speed.as_integer_ratio() # 3 / 2 = > num 3 , den 2 ;  4 1 

    pitch_zones = []
    unvoiced_zones = []
    for reg_number, reg in enumerate(regions):
        start,end = reg
        if is_frame_speech[start]:
            pitch_centered = []
            if pitch_version == "pitch_variable": # pitch variable
                is_first_time = True
                p_samples = find_max_probability_f0(reg_number,start,end,indexes,f0_in_samples)
                peaks, _ = find_peaks(x, distance = p_samples - p_samples*percent)
                for i,peak in enumerate(peaks):
                    if start <= peak and peak <= end:
                        f0_idx = search_pitch_samples(peak, indexes) # busco cuanto pitch tiene este peak
                        curr_pitch = f0_in_samples[f0_idx]
                        curr_pitch = int(curr_pitch*percent_pitch)
                        w_sonora = np.hanning(2*curr_pitch+1)
                        if peak-curr_pitch>=0 and peak+curr_pitch <= len(x) - 1: 
                            samples_windowed = x[peak-curr_pitch:peak+curr_pitch+1]*w_sonora
                            pitch_centered.append(pitch_centered_segment(curr_pitch, peak, samples_windowed))
                            pitch_zones.append([peak-curr_pitch,peak+curr_pitch])
                        
            #-ASI ESTABA ANTES CON FLANGE Y EL OTRO CAMBIO:
            if pitch_version == "pitch_cte": # pitch cte
                is_first_time = True
                p_samples = find_max_probability_f0(reg_number,start,end,indexes,f0_in_samples)
                p_samples = int(percent_pitch*p_samples) 
                peaks, _ = find_peaks(x, distance = p_samples - p_samples*percent)
                w_sonora = np.hanning(2*p_samples+1)
                
                for i,peak in enumerate(peaks):
                    if start <= peak and peak <= end:
                        if peak-p_samples>=0 and peak+p_samples <= len(x) - 1: 
                            samples_windowed = x[peak-p_samples:peak+p_samples+1]*w_sonora
                            pitch_centered.append(pitch_centered_segment(p_samples, peak, samples_windowed))
                            pitch_zones.append([peak-p_samples,peak+p_samples])

            for i, centered in enumerate(pitch_centered):
                t_ = ceil(centered.t * (1/speed))
                
                if speed == 1 or speed == 2 or speed == 4:
                    if den > i%num: #  
                        if t_-centered.pitch>=0 and t_+centered.pitch <= len(new_audio) - 1: 
                            new_audio[t_-centered.pitch:t_+centered.pitch+1] += centered.samples_windowed
                            pitch_zones.append([t_-centered.pitch,t_+centered.pitch])
                            
                elif speed == 1.25 or speed == 1.5 or speed == 1.75:
                    if pitch_dif_mode == "1": # fijo primer t_ y paso la dif del tercero al segundo (no actualizo cada 3)
                        if is_first_time: # si es 1.5 => 3, 2 
                            curr_t_ = t_ # cada 3 actualizo el t_
                        if den> i%num:
                            if is_first_time:
                                distance = 0
                            else:
                                distance += pitch_centered[i].t - pitch_centered[i-1].t

                            curr_t_shifted = curr_t_+ distance 
                            if curr_t_shifted-centered.pitch>= 0 and curr_t_shifted+centered.pitch < len(new_audio):
                                new_audio[curr_t_shifted-centered.pitch:curr_t_shifted+centered.pitch+1] += centered.samples_windowed
                                pitch_zones.append([t_-centered.pitch,t_+centered.pitch])
                        
                        if is_first_time:
                            is_first_time = False
                    if pitch_dif_mode == "2":
                        if i%num == 0: # si es 1.5 => 3, 2 
                            curr_t_ = t_ # cada 3 actualizo el t_
                            distance = 0 # reseteo la distancia

                        if den> i%num:
                            if i%num == 0:
                                distance = 0
                            else:
                                distance += pitch_centered[i].t - pitch_centered[i-1].t

                            curr_t_shifted = curr_t_+ distance 
                            if curr_t_shifted-centered.pitch>= 0 and curr_t_shifted+centered.pitch < len(new_audio):
                                new_audio[curr_t_shifted-centered.pitch:curr_t_shifted+centered.pitch+1] += centered.samples_windowed
    
        else:# si es no sonora
            #w_no_sonora = np.hanning(2*unvoiced_samples+1)
            #samp = int(end-start)
            i = 0
            leap = unvoiced_samples
            unvoiced_centered = []
            right_lim = start
            while right_lim  < end:
                middle = leap 
                t = start + middle + i*leap
                right_lim = t + leap
                curr_w = x[t - leap : t + leap + 1]
                w_n_s = np.hanning(len(curr_w))
                unvoiced_centered.append(pitch_centered_segment(leap,t,curr_w*w_n_s))
                i += 1

            for i,unv_cent in enumerate(unvoiced_centered): 
                t_ = ceil(unv_cent.t*(1/speed))
                if speed == 1.0 or speed == 2.0 or speed == 4.0:
                    if den > i%num:
                        if t_-unv_cent.pitch>=0 and t_+unv_cent.pitch < len(new_audio):
                                new_audio[t_-unv_cent.pitch:t_+unv_cent.pitch+1] += unv_cent.samples_windowed
                                unvoiced_zones.append([t_-unv_cent.pitch,t_+unv_cent.pitch])

                elif speed == 1.25 or speed == 1.5 or speed == 1.75:
                    if i%num == 0: # si es 1.5 => 3, 2 
                            curr_t_ = t_ # cada 3 actualizo el t_
                            distance = 0 # reseteo la distancia
                    if den> i%num:
                        if i%num == 0:
                            distance = 0
                        else:
                            distance += unvoiced_centered[i].t - unvoiced_centered[i-1].t

                        curr_t_shifted = curr_t_+ distance 
                        if curr_t_shifted-unv_cent.pitch>= 0 and curr_t_shifted+unv_cent.pitch < len(new_audio):
                            new_audio[curr_t_shifted-unv_cent.pitch:curr_t_shifted+unv_cent.pitch+1] += unv_cent.samples_windowed
                            unvoiced_zones.append([t_-unv_cent.pitch,t_+unv_cent.pitch])

                    # if i%num == 0: # si es 1.5 => 3, 2 
                    #     curr_t_ = t_ # cada 3 actualizo el t_
                    # if den> i%num:
                    #     curr_t_shifted = curr_t_+(i%num)*unvoiced_samples 
                    #     if curr_t_shifted-unvoiced_samples>=0 and curr_t_shifted+unvoiced_samples < len(new_audio):
                    #             new_audio[curr_t_shifted-unvoiced_samples:curr_t_shifted+unvoiced_samples+1] += unv_cent.samples_windowed
                    #             unvoiced_zones.append([curr_t_shifted-unvoiced_samples,curr_t_shifted+unvoiced_samples])
    return new_audio, pitch_zones, unvoiced_zones