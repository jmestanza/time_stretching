import librosa 
import scipy
import os 

from detect_pitch import get_fundamental_frequency

os.chdir("../data")

fs = 8000
ahh, _ = librosa.load("ahh_without_silences_8k.wav", sr=fs)

y = ahh[:1000]
a = librosa.lpc(y, 12)
e_ = scipy.signal.lfilter(a,[1], y) # obtenemos la senial error a traves del filtrado

T = 20*1e-3
samples = int(T * fs) 
L = samples//4 
K = L

# mejores resultados hasta ahora con L = samples //4 y K = L
f0,sample_f0 = get_fundamental_frequency(e_,K,L,fs)
print(sample_f0)