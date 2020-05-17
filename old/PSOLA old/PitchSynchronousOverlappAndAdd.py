import librosa 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import find_peaks

fs = 8000
ahh, _ = librosa.load("ahh_without_silences_8k.wav", sr=fs)

M = 100
overlap_percent = 0.9
overlap = int(overlap_percent*M) # 50%

shaped_audio = ahh[:len(ahh)//M * M]
w = np.hamming(M)

new_audio = np.zeros(len(shaped_audio))

x = shaped_audio
peaks, _ = find_peaks(x, height=0, distance = 50)
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.plot(np.zeros_like(x), "--", color="gray")
plt.show()

pitch_samples = 60
# print(len(x[peaks[1]-60:peaks[1]+60+1])) # 60+60+1
pitch_centered_peaks = []
for i,peak in enumerate(peaks):
    if i !=0  and i != len(peaks) -1 :
        peak_partition = x[peak-pitch_samples:peak+pitch_samples+1]
        window = np.hamming(len(peak_partition))
        pitch_centered = peak_partition*window
        # hago mas grande el array (hago zero padding en los costados por ahora en realidad esta mal)
        pitch_centered_peaks.append([ peak-pitch_samples , pitch_centered])
        # pitch_centered_peaks[i,0] = 
        # pitch_centered_peaks[i,1] = pitch_centered

alpha = 0
for index, analysis_window in pitch_centered_peaks:
    new_audio[alpha+index:alpha+index+2*pitch_samples+1] += analysis_window
    #alpha -= 30 equivale a un x2 pero no se escucha bien
    alpha -=20
    # como que hay un cambio en el pitch :/

#si quiero ir mas rapido, la distancia entre pitches dberia ser mas chica!
librosa.output.write_wav("new_audio.wav", new_audio, fs)