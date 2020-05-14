import librosa
from environment.modules.analysislab import frameanalisys
import numpy as np
from environment.modules.analysislab.user_interface import featuresnames, Fs, hopsize


# Spectral Flatness
def compute_flatness(audio):
    flatness = librosa.feature.spectral_flatness(y=audio,
                                                 S=None,
                                                 n_fft=2048,
                                                 hop_length=hopsize(),
                                                 amin=1e-10,
                                                 power=2.0)
    return np.nanmean(np.trim_zeros(flatness[0]))


# Spectral Rolloff
def compute_rolloff(audio):
    rolloff = librosa.feature.spectral_rolloff(y=audio,
                                               sr=Fs(),
                                               S=None,
                                               n_fft=2048,
                                               hop_length=hopsize(),
                                               freq=None,
                                               roll_percent=0.85)
    return np.nanmean(np.trim_zeros(rolloff[0]))


def compute_spectral_centroid(audio):
    return librosa.feature.spectral_centroid(y=audio,
                                             sr=Fs(),
                                             S=None,
                                             n_fft=2048,
                                             hop_length=hopsize(),
                                             freq=None, )


def average_centroid(audio):
    centroid = compute_spectral_centroid(audio)
    return np.average(centroid)


def spectral_bandwith(audio):
    return np.average(librosa.feature.spectral_bandwidth(y=audio,
                                                         S=None,
                                                         n_fft=2048,
                                                         hop_length=hopsize(),
                                                         freq=None, ))


def zero_crossing(audio):
    return np.average(librosa.feature.zero_crossing_rate(y=audio,
                                                         hop_length=hopsize(), ))


def mfccs(audio):
    return librosa.feature.mfcc(y=audio,
                                S=None,
                                n_fft=2048,
                                hop_length=hopsize(),
                                )


def getfeatures(audio):
    featurearray = np.zeros(len(featuresnames()))
    if np.amax(audio) != 0:
        audio = audio * loudness() / np.amax(audio)
    featurearray[0] = compute_flatness(audio)
    featurearray[1] = compute_rolloff(audio)
    featurearray[2] = average_centroid(audio)
    featurearray[3] = spectral_bandwith(audio)
    featurearray[4] = zero_crossing(audio)
    a = mfccs(audio)
    mffcoeff = np.average(a, axis=1)

    framedata = frameanalisys.getframefeatures(audio)
    featurearray[5] = framedata[0]
    featurearray[6] = framedata[1]
    featurearray[7] = framedata[2]

    for i in np.arange(mffcoeff.shape[0]):
        featurearray[-1 - i] = mffcoeff[-1 - i]
    return featurearray
