import librosa 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

def plot_autocorrelation_and_signal(partition,g):
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(partition)
    plt.legend(["senal"])
    plt.subplot(122)
    plt.plot(g)
    plt.legend(["autocorrelation de la senal"])
    plt.suptitle('autocorrelation testing')
    plt.show()

def get_samples_for_process(audio, K, L):
    side_len = int(K+L-1)
    N = len(audio) - 2*side_len
    middle_L_cnt = int(N // L)
    middle_L = middle_L_cnt * L # voy a saltar de a L
    shaped_audio = audio[:side_len + middle_L + side_len]
    # ahora q tengo el shaped_audio, solo falta dividir en samples
    total_sz = 2*side_len+L
    cut_audios = np.zeros((middle_L_cnt,total_sz))
    indexes = []
    for i in range(middle_L_cnt):
        cut_audios[i] = shaped_audio[i*L:i*L+total_sz]
        indexes.append(side_len+i*L)

    return cut_audios, indexes

def process_window_autocorrelation(x,K,L,window_type):
    if len(x) != (2*(K+L-1)+L):
        return None 
    if window_type == "hanning":
        w1 = np.hanning(L) # da muy bien con hanning!!
        w2 = np.hanning(L+K)
    elif window_type == "rectangular":
        w1 = np.ones(L)
        w2 = np.ones(L+K) 
    
    x_middle = x[K+L-1:K+L-1+L] * w1
    x1_fixed = np.hstack((np.zeros(K+L-1),x_middle,np.zeros(K+L-1)))

    gamma = []
    for cnt in range(K+2*L-1):
        x2_fixed = np.zeros(len(x1_fixed))
        x2_moving = x[cnt:cnt+K+L] * w2
        x2_fixed[cnt:cnt+K+L] = x2_moving # este se va moviendo
        gamma.append(np.sum(x1_fixed*x2_fixed))
        cnt += 1
    return gamma

def get_fundamental_frequency(audio, is_frame_speech,K,L,fs, w_type = "hanning", show_demo = False):
    audio = audio[:len(is_frame_speech)]
    #split_audio, indexes = get_samples_for_process(audio, K, L)
    split_audio, indexes = get_samples_for_process(audio,2*K,2*L)

    f0s_in_samples = []
    for i, partition in enumerate(split_audio):
#        autocorrelation = process_window_autocorrelation(partition, K, L, w_type)
        full_autocorr = process_window_autocorrelation(partition,2*K,2*L,w_type)
        autocorrelation = full_autocorr[len(full_autocorr)//2:]

        if show_demo:
            plot_autocorrelation_and_signal(partition,autocorrelation)
            show_demo = False
        if is_frame_speech[indexes[i]]:
            # puede que esto valga 0 en algun momento, cuidado
            f0_hat_in_samples = np.argmax(autocorrelation) 
        else:# si no es sonoro, lo computo como 0
            f0_hat_in_samples = 0
        f0s_in_samples.append(f0_hat_in_samples)

    return indexes, f0s_in_samples



# x = np.ones(1000)

# L = 50
# K = 25
# #cut_audios, indexes = get_samples_for_process(x,2*K,2*L)
# full_autocorr = process_window_autocorrelation(cut_audios[0],2*K,2*L,"rectangular")
# autocorr = full_autocorr[len(full_autocorr)//2:]
# plt.plot(autocorr)
# # mi K es K = 12.5
# plt.show()
