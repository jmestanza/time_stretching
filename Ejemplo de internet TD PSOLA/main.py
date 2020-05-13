from td_psola import *

orig_signal, fs = librosa.load("female_scale.wav", sr=44100)
N = len(orig_signal)
# Pitch shift amount as a ratio
f_ratio = 2 ** (-2 / 12)
new_signal = shift_pitch(orig_signal, fs, f_ratio)
plt.style.use('ggplot')
plt.plot(orig_signal[:-1])
plt.show()
plt.plot(new_signal[:-1])
plt.show()
# Write to disk
librosa.output.write_wav("female_scale_transposed_{:01.2f}.wav".format(f_ratio), new_signal, fs)