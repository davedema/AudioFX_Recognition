import librosa
import numpy as np
import scipy as sp
from executables.modules.glob import Fs


# Computes Spectral Flatness
def compute_flatness(audio):
    flatness = librosa.feature.spectral_flatness(y=audio,
                                                 S=None,
                                                 n_fft=2048,
                                                 hop_length=512,
                                                 amin=1e-10,
                                                 power=2.0)
    return flatness


# Computes Spectral Rolloff
def compute_rolloff(audio):
    rolloff = librosa.feature.spectral_rolloff(y=audio,
                                               sr=22050,
                                               S=None,
                                               n_fft=2048,
                                               hop_length=512,
                                               freq=None,
                                               roll_percent=0.85)
    return rolloff


# Those two functions define a Butterworth low pass filter
def butter_lowpass_filter(data, cutoff, Fs, order=5):
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq
    b, a = sp.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = sp.signal.lfilter(b, a, data)
    return y


# Gets average over sections of the array
def linear(waveform, n):
    averaged_section = np.array([])
    section_length = (int)((len(waveform) - 1) / n)
    lin_fx = np.zeros(len(waveform))

    for i in np.arange(n):

        wv_section = waveform[i * section_length: (i + 1) * section_length]
        averaged_section = np.append(averaged_section, np.average(wv_section))
        index1 = i * (section_length - 1)
        index2 = (i + 1) * (section_length - 1)

        if i == 0:
            linear = np.linspace(0, averaged_section[i], section_length, endpoint=False)
        if i > 0:
            linear = np.linspace(averaged_section[i - 1], averaged_section[i], section_length, endpoint=False)

        lin_fx[index1:index2] = linear[0:index2 - index1]

    return lin_fx


# Gets normalized average of the array from index a to b
def get_indexed_average(frame, a, b):
    index1 = int(a * (len(frame) - 1))
    index2 = int(b * (len(frame) - 1))
    win_frame = np.abs(frame[index1:index2])
    value = np.average(win_frame)

    return value


# Gets autocorrelation of the array
def autocorrelate(x):
    autocorr = np.correlate(x, x, mode='full')[len(x) - 1:]
    module_autocorr = np.abs(autocorr)
    norm_autocorr = module_autocorr / np.amax(module_autocorr)
    return norm_autocorr


# Feature that discriminates btw tremolo, nofx, distortion
# obtained by doing the autocorrelation of the subtraction btw filtered waveform and over sections linearized waveform
# afterwards it counts the number of relative maxima in the autocorrelation of the difference
def tremolo_feature_2(audio_waveform, a, b):
    filtered_wv = butter_lowpass_filter(audio_waveform, 1000, Fs(), order=3)
    filtered_wv = np.trim_zeros(filtered_wv)
    maxPos = sp.signal.argrelextrema(filtered_wv, np.greater)
    n_sections = int(len(maxPos[0]))

    linear_fwv = linear(filtered_wv, n_sections)
    diff_fwv = filtered_wv - linear_fwv
    result = get_indexed_average(diff_fwv, a, b)

    return result


def tremolo_feature(audio_waveform):
    filtered_wv = butter_lowpass_filter(audio_waveform, 1000, Fs(), order=3)
    filtered_wv = np.trim_zeros(filtered_wv)
    maxPos = sp.signal.argrelextrema(filtered_wv, np.greater)
    n_sections = int(len(maxPos[0]))

    linear_fwv = linear(filtered_wv, n_sections)
    diff_fwv = filtered_wv - linear_fwv
    autocorrelation = autocorrelate(diff_fwv)

    maxPos_auto = sp.signal.argrelextrema(autocorrelation, np.greater)
    n_rel_max = int(len(maxPos_auto[0]))

    return n_rel_max