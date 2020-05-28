def LPC_short_time_filter(audio_f,audio_noised,window_size, apply_hanning = True):
    leap = window_size // 2
    lpc_coeffs = []
    cnt_leaps = len(audio_f) // leap 
    error = np.zeros(len(audio_f))

    for i in range(cnt_leaps):
        curr_rw = audio_noised[i*leap : (i+1)*leap+leap] # overlap de 50%
        if apply_hanning:
            hanning_w = np.hanning(len(curr_rw))
            curr_rw *= hanning_w
        a = librosa.lpc(curr_rw, 12)
        lpc_coeffs.append(a)
        error[i*leap : (i+1)*leap+leap] += scipy.signal.lfilter(a, [1], audio_f[i*leap : (i+1)*leap+leap])
    
    return error, lpc_coeffs

def LPC_short_time_defilter(psola_err, speed, lpc_coeffs, window_size, apply_hanning = True):
    leap = window_size // 2
    cnt_leaps = len(psola_err) // leap 
    final_audio = np.zeros(len(psola_err))
    num, den = speed.as_integer_ratio() # 3 / 2 = > num 3 , den 2 ;  4 1 

    new_lpc_coeffs = []
    for i in range(len(lpc_coeffs)):
        if den> i%num:
            new_lpc_coeffs.append(lpc_coeffs[i])

    for i in range(cnt_leaps):
        curr_rw = psola_err[i*leap : (i+1)*leap+leap] # overlap de 50%
        if apply_hanning:
            hanning_w = np.hanning(len(curr_rw))
            curr_rw *= hanning_w
        final_audio[i*leap : (i+1)*leap+leap] += scipy.signal.lfilter([1], new_lpc_coeffs[i], curr_rw)
    return final_audio
def speed_up_with_LPC_ST(audio_path, speed, sample_rate, T=20*1e-3):
    audio_f, audio_i = get_float_and_int_audio(audio_path,fs, cut_audio = 4) # audio_i lo necesita el VAD
    audio_noised = get_noise_for_lpc(audio_f, std=5e-3) 

    K , L = get_K_and_L(T,fs)

    speech, not_speech, is_voiced, frame_len = vad_config(audio_i, fs)
    
    regions = separate_regions(is_voiced)
    regions_lpc = regions.copy()

    window_size = int(75e-3*fs)
    error, lpc_coeffs = LPC_short_time_filter(audio_f,audio_noised,window_size, apply_hanning = True)
    
    percent_pitch = 0.6
    psola_error, f0_idx, f0, pitch_zones, unvoiced_zones =  process_error(error, K, L, fs,0.5,percent_pitch,speed,is_voiced,regions)
    
    voiced_filt =  LPC_short_time_defilter(psola_error, speed, lpc_coeffs, window_size, apply_hanning = True)

    return voiced_filt, psola_error, error, audio_f , is_voiced

speed = 2.0
audio_final, psola_err, err, audio_original, is_voiced = speed_up_with_LPC_ST(audio, speed, fs)
plot_four_signals(audio_original,audio_final,psola_err,is_voiced,fs,speed)
Audio(data=audio_final, rate=fs)