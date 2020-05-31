import matplotlib.pyplot as plt 
from vad import float2pcm, pcm2float, frame_generator, join_vad_frames, get_vad_frames
from vad import Frame
from scipy import signal
import matplotlib.patches as mpatches
import librosa 
import scipy
import numpy as np
import math
import webrtcvad

def plot_voiced_regs(regions, is_frame_speech, audio_f):
  plt.figure(figsize=(20, 5))
  green_patch = mpatches.Patch(color='green', label='sonoro')
  red_patch = mpatches.Patch(color='red', label='no-sonoro')
  plt.legend(handles=[green_patch, red_patch])
  plt.xlabel("muestras")
  plt.ylabel("amplitud")

  #ploteo regiones sonoras y no sonoras
  for i,reg in enumerate(regions):
    start,end = reg  
    if start < 1000000 and end < 1000000:
      if(is_frame_speech[start] == True):
        # plt.vlines(x=start, ymin=0, ymax=1.0)
        plt.hlines(y=0.65, xmin=start, xmax=end, linewidth=3, colors='g')
        # plt.vlines(x=end, ymin=0, ymax=1.0)
      else:
        plt.hlines(y=0.65, xmin=start, xmax=end, linewidth=3, colors='r')

    
  plt.plot(audio_f[:1000000])
  plt.grid()
  plt.figure()
# REGIONS BUGGEADO CON EL CUAL DABA BUENOS RESULTADOS
# def separate_regions(is_voiced):
#     regions = []
#     for i in range(len(is_voiced)):
#         if i == len(is_voiced) - 1:
#             regions.append([start,i])
#         elif i ==0:
#             start = 0
#     else:
#         curr_region = is_voiced[i] # si es sonoro o no sonoro
#         if curr_region != is_voiced[i-1]:
#             regions.append([start,i-1]) # pusheo la region anterior
#         start = i
#     return regions
def separate_regions(is_voiced):
    regions = []
    for i in range(len(is_voiced)):
        curr = is_voiced[i]
        if i == 0: 
            start = 0
            last = is_voiced[i] # asigno last como en el que estoy
            # si son iguales, no se hace nada y se prosigue
        else:
            last = is_voiced[i-1]
            if curr != last: # si cambio algo, pongo nuevo start
                regions.append([start,i-1])
                start = i  
            # suponiendo q ya flusheo el ultimo arreglo,
            # ya sea inmediato anterior el ultimo pedazo de arreglo
            # o antes...
            if i == len(is_voiced) - 1:
                regions.append([start,i]) 
                # esto funciona si no cambio, last != curr, tengo guardado start
                # y si cambio tambien [start= i, i]

        #print("curr",curr,"last",last, "es distinto", curr!=last, "start", start)
    return regions

def get_float_and_int_audio(audio_path,fs, cut_audio = None):
    audio_f, _ = librosa.load(audio_path, sr=fs, mono=True)
    audio_i = float2pcm(audio_f, dtype="int32")
    if cut_audio != None:
        audio_f = audio_f[:int(cut_audio*fs)]
        audio_i = audio_i[:int(cut_audio*fs)]
    return audio_f, audio_i

def vad_config(audio_i, fs):
    vad = webrtcvad.Vad(3)    #0 is the least aggressive about filtering out non-speech, 3 is the most aggressive; 3 pone mas frames en no-sonoros
    #divido el audio en frames de 20 milisegundos
    frames = list(frame_generator(20, audio_i, fs))
    speech, not_speech, is_frame_speech, frame_len = get_vad_frames(frames, vad, fs)
    return speech, not_speech, is_frame_speech, frame_len

def get_regions_in_new_time(regions,speed, mode): 
    sub_regions = []
    for i,reg in enumerate(regions):
        start,end = reg
        if i == 0: # si es el primero, inicia en 0, 
            new_start = 0
        else:  # sino empieza en el anterior final + 1
            new_start = last_end + 1
            
        if mode == "ceil":
            new_end = math.ceil(end*(1/speed))
        elif mode == "floor":
            new_end = math.floor(end*(1/speed))
        else:
            return None

        sub_regions.append([new_start,new_end])
        last_end = new_end
    return sub_regions


def get_K_and_L(T,fs):
    samples = int(T * fs) 
    L = samples//2
    K = L
    print("periodicidad minima que se puede estimar",fs/K, "Hz")
    return K, L 

def plot_four_signals(audio_original,audio_procesado,error_psola,is_voice,fs, speed, figsize_ = (15,6),normalize = False, fill_with_zeros= False):
    if normalize:
     audio_original = librosa.util.normalize(audio_original)
     audio_procesado = librosa.util.normalize(audio_procesado)
    if fill_with_zeros:
        if speed != 1.0:
            audio_procesado = np.hstack((audio_procesado,np.zeros(len(audio_original)-len(audio_procesado))))
            error_psola = np.hstack((error_psola,np.zeros(len(audio_original)-len(error_psola))))
            is_voice = np.hstack((is_voice,np.zeros(len(audio_original)-len(is_voice))))
            
    time = np.linspace(0, len(audio_original)//fs, len(audio_original))
    fig = plt.figure(figsize=figsize_)
    ax = fig.add_subplot(411)
    ax1 = fig.add_subplot(412)
    ax.set_title('Audio en inglés x{}'.format(speed))
    ax.plot(time, audio_original, 'b') 
    ax1.plot(time[:len(audio_procesado)],audio_procesado, 'g')
    ax1.set_xlabel("tiempo (s)")
    ax.set_ylabel("speed = 1.0")
    ax1.set_ylabel('speed = {}'.format(speed))

    ax2 = fig.add_subplot(413)
    ax2.set_xlabel("tiempo (s)")
    ax2.set_ylabel('error'.format(speed))
    ax2.plot(time[:len(error_psola)],error_psola, "r")

    ax3 = fig.add_subplot(414)
    ax3.set_xlabel("tiempo (s)")
    ax3.set_ylabel('is voiced'.format(speed))
    ax3.plot(time[:len(is_voice)],is_voice, "y")
    
    plt.show()

def draw_windows(zones,regions,reg_num,cnt_ventanas, xlabel, title): # pitch/unvoiced-region and number of windows to plot
    start_reg = regions[reg_num][0]
    end_reg = regions[reg_num][1]
    suma = np.zeros(end_reg-start_reg)
    
    for i,reg in enumerate(zones):
        start,end = reg
        if start_reg <= start and end <= end_reg and i < cnt_ventanas:
            y = np.zeros(len(suma))
            w = np.hanning(end-start)
            y[start-start_reg:end-end_reg] = w
            suma[start-start_reg:end-start_reg] += w
            t = [ el + start_reg for el in list(range(len(y)))]
            plt.plot(t,y,"r")
    plt.plot(t,suma)
    plt.title(title)
    plt.xlabel(xlabel)