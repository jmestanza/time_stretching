import librosa 
import os
from detect_pitch import get_fundamental_frequency
from modified_psola import modified_psola

fs = 8000

os.chdir("../data")

audio_file = "ahh_without_silences_8k_norm"
ahh, _ = librosa.load(audio_file+".wav", sr=fs)

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

f0, sample_f0 = get_fundamental_frequency(ahh,K,L,fs)
print("Frecuencia fundamental : ",f0," ", sample_f0)
speed = 1
new_audio = modified_psola(ahh,sample_f0, speed)

os.chdir("../output")
librosa.output.write_wav(audio_file+f"_speed_x{speed}.wav", new_audio, fs)
