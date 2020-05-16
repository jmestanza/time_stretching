import librosa 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read


def get_samples_for_process(audio, K, L):
    side_len = int(K+L-1)
    N = len(audio) - 2*side_len
    middle_L_cnt = int(N // L)
    middle_L = middle_L_cnt * L # voy a saltar de a L
    shaped_audio = audio[:side_len + middle_L + side_len]
    # ahora q tengo el shaped_audio, solo falta dividir en samples
    total_sz = 2*side_len+L
    cut_audios = np.zeros((middle_L_cnt,total_sz))
    for i in range(middle_L_cnt):
        cut_audios[i] = shaped_audio[i*L:i*L+total_sz]
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


# esto es para una ventana
L = 10
K = 5
#short2 = ahh[:2*(K+L-1)+L] # esto esta centrado en x[K+L-1:K+L-1+L] (tamanio L pero offset K+L-1 y ultimo pedazo de K+L-1)

#plt.plot(ahh[:100])
# a 8Khz tengo 100 muestras => 80Hz
fmin = 50
fmax = 500

T = 20e-3
#L = int(T*fs) # muestras de la ventana
#K = L//2
L = 25 # muestras de la ventana
K = 10

split_audio = get_samples_for_process(ahh, K, L)
min_f_arr = []
for partition in split_audio:
    g = process_window_amdf(partition, K, L, "hamming")
    min_f_sample = np.argmin(partition)
    if min_f_sample == 0:
        continue
    min_f = fs/min_f_sample # fs / Muestras = f0
    if min_f > 500:
        continue
    min_f_arr.append(min_f)
    #print(min_f)

hist, bin_edges = np.histogram(np.array(min_f_arr), bins=150)  
print("La fundamental por promedio es: ",np.mean(np.array(min_f_arr)))
print("La fundamental con mas probabilidad :",bin_edges[np.argmax(hist)])

plt.show()


# f = plt.figure(figsize=(10, 4))
# plt.subplot(121)
# plt.plot(partition)
# plt.legend(["senal"])
# plt.subplot(122)
# plt.plot(g)
# plt.legend(["AMDF de la senal"])
# plt.suptitle('Amdf testing')
# plt.show()



