import librosa 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

fs = 8000
ahh, _ = librosa.load("ahh_without_silences_8k.wav", sr=fs)
pitch_samples = 60

def plot_peaks(x,peaks):
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
    plt.show()

#alpha_factor = -30 equivale a un x2 pero hay un cambio en el pitch :/
#alpha_factor = -20 equivale a un x1.5 mas o menos pero cambia un poco el pitch

def PSOLA(x,pitch_samples,alpha_factor):
    new_audio = np.zeros(len(x))
    peaks, _ = find_peaks(x, height=0, distance = pitch_samples-int(0.16*pitch_samples))
    #plot_peaks(x,peaks) # debugging
    
    pitch_centered_peaks = []
    w = np.hamming(2*pitch_samples+1)
    
    for i,peak in enumerate(peaks):
        if i !=0  and i != len(peaks) -1 :
            pitch_centered_samples = x[peak-pitch_samples:peak+pitch_samples+1]
            pitch_centered_windowed = pitch_centered_samples*w
            # hago mas grande el array (hago zero padding en los costados por ahora en realidad esta mal)
            pitch_centered_peaks.append([ peak-pitch_samples , pitch_centered_windowed])

    alpha = 0
    for index, analysis_window in pitch_centered_peaks:
        new_audio[alpha+index:alpha+index+2*pitch_samples+1] += analysis_window
        alpha += alpha_factor
        
    #si quiero ir mas rapido, la distancia entre pitches dberia ser mas chica!
    librosa.output.write_wav(f"audio_f0_sample_{pitch_samples}_alpha_{alpha_factor}.wav", new_audio, fs)