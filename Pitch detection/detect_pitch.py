import librosa 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read


def get_samples_for_process(audio, K, L):
    side_len = K+L-1
    N = len(audio) - 2*side_len
    middle_L = (N // L) * L # voy a saltar de a L
    shaped_audio = audio[side_len + middle_L + side_len:]
    # ahora q tengo el shaped_audio, solo falta dividir en samples
    cut_audios = np.zeros((middle_L,2*side_len+L))
    for i in range(middle_L):
        cut_audios[i] = shaped_audio[i*L:i*L+2*side_len+L]
    return cut_audios

def process_window_amdf(x,K,L,window_type):
    if len(x) != (2*(K+L-1)+L):
        return None 
    if window_type == "hamming":
        w1 = np.hamming(L) # da muy bien con hamming!!
        w2 = np.hamming(L+K)
    elif window_type == "rectangular":
        w1 = np.ones(L) #np.hamming(len(short)) # afeura de esto vale 0
        w2 = np.ones(L+K) #np.hamming(len(short)) # afuera vale 0
    
    x_middle = x[K+L-1:K+L-1+L] * w1
    x1_fixed = np.hstack((np.zeros(K+L-1),x_middle,np.zeros(K+L-1)))

    gamma = []
    for cnt in range(K+2*L-1):
        x2_fixed = np.zeros(len(x1_fixed))
        x2_moving = x[cnt:cnt+K+L] * w2
        x2_fixed[cnt:cnt+K+L] = x2_moving # este se va moviendo
        gamma.append(np.sum(np.abs(x1_fixed-x2_fixed)))
        cnt += 1
    return gamma

fs = 8000
ahh, _ = librosa.load("ahh_without_silences_8k.wav", sr=fs)
shh, _ = librosa.load("shh_without_silences_8k.wav", sr=fs)

#plt.plot(ahh[0:100])
#plt.show()

# esto es para una ventana
L = 10
K = 5
short2 = ahh[:2*(K+L-1)+L] # esto esta centrado en x[K+L-1:K+L-1+L] (tamanio L pero offset K+L-1 y ultimo pedazo de K+L-1)

g = process_window_amdf(short2,K,L, "hamming")
plt.plot(short2)
plt.plot(g)
plt.legend(["senal","AMDF de la senal"])
plt.show()