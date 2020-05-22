
import webrtcvad
import numpy as np


def get_vad_frames(frames, vad, fs):
  """ Parametro: lista de frames generados con frame_generator()
      Devuelve lista speech con los frames sonoros concatenados, otra not_speech con los no-sonoros concatenados, y
      y lista is_frame_speech para saber si el frame original es sonoro o no (True=sonoro) 
  """
  speech = list()
  not_speech = list()
  is_frame_speech = list()

  for frame in frames:
    if vad.is_speech(frame.bytes, fs):     #si el frame que analizo es sonoro
      speech.extend(frame.bytes)                    #lo guardo en la lista correspondiente y actualizo el indice
      is_frame_speech.append(True)
    else:                                     #si el frame que analicé no era sonoro
      not_speech.extend(frame.bytes)                #lo guardo en lista de no-sonoros
      is_frame_speech.append(False)

  speech = pcm2float(sig=np.array(speech))

  not_speech = pcm2float(sig=np.array(not_speech))

  return speech, not_speech, is_frame_speech



def join_vad_frames(voiced, not_voiced, indexes):
  """genera un arreglo con los tramos sonoros y no sonoros post psola treatment
  voiced = lista con tramos sonoros
  not_voiced = lista con tramos no-sonoros
  indexes = lista detallando qué frame es sonoro y cuál no
  
  returns: array
  """
  audio = list()
  for is_speech in indexes:
    if is_speech == True:   #si el frame es sonoro
      if(voiced):
        audio.append(voiced[0].copy())
        voiced.pop(0)
    else:                   #si el frame no es sonoro lo anexo y lo borro
      if(not_voiced):
        audio.append(not_voiced[0].copy())
        not_voiced.pop(0)

  final = list()
  for aud in audio:
    final.extend(aud)

  return final

#https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

#https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
def pcm2float(sig, dtype='float64'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

#https://github.com/wiseman/py-webrtcvad/blob/3b39545dbb026d998bf407f1cb86e0ed6192a5a6/example.py
class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


#https://github.com/wiseman/py-webrtcvad/blob/3b39545dbb026d998bf407f1cb86e0ed6192a5a6/example.py
def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

