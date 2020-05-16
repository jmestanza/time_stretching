import librosa 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import scipy
import numpy as np

from DetectPitch import *

import matplotlib.pyplot as plt
def plot_comparison(y1,y2,name1,name2,title):
    plt.plot(y1)
    plt.plot(y2, linestyle='--')
    plt.legend([name1, name2])
    plt.title(title)

fs = 8000
ahh, _ = librosa.load("ahh_without_silences_8k.wav", sr=fs)

y = ahh[:1000]
a = librosa.lpc(y, 12)
y_hat = scipy.signal.lfilter([0] + -1*a[1:] , [1], y)
e_ = scipy.signal.lfilter(a,[1], y) # obtenemos la senial error a traves del filtrado

f = plt.figure(figsize=(10, 4))
plt.subplot(121)
# plt.plot(y)
# plt.plot(y_hat)
# plt.legend(["y","y_hat"])
# plt.subplot(122)
# plt.plot(e_)
# plt.legend(["error"])
# plt.suptitle('signal and error obtained by filtering with LPC')
# plt.show()

T = 20*1e-3
samples = int(T * fs) 
L = samples//4 # esto esta justificado arriba
K = L

# mejores resultados hasta ahora con L = samples //4 y K = L


f0,sample_f0 = get_fundamental_frequency(e_,K,L,fs)