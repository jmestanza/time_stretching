import librosa 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

from DetectPitch import *
from PitchSynchronousOverlappAndAdd import *

fs = 8000
ahh, _ = librosa.load("ahh_without_silences_8k.wav", sr=fs)

# necesito al menos dos periodos de la senial
#200 = (50+25)+50+(50+25)
#L = 50 # muestras de la ventana
#K = L//2
# tamanio total por observacion
# total = 4*L

#quiero tener 20 milisegundos de senial

T = 20*1e-3
samples = int(T * fs) 
L = samples//4
K = L

f0,sample_f0 = get_fundamental_frequency(ahh,K,L,fs)
print("Frecuencia fundamental : ",f0," ", sample_f0)
PSOLA(ahh,sample_f0,-20)
# -20 = x1.5
