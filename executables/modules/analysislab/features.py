import librosa
from executables.modules.analysislab import frameanalisys
import numpy as np
from executables.modules.analysislab.user_interface import featuresnames, Fs


# Spectral Flatness
def compute_flatness(audio):
    flatness = librosa.feature.spectral_flatness(y=audio,
                                                 S=None,
                                                 n_fft=2048,
                                                 hop_length=512,
                                                 amin=1e-10,
                                                 power=2.0)
    return np.nanmean(np.trim_zeros(flatness[0]))


# Spectral Rolloff
def compute_rolloff(audio):
    rolloff = librosa.feature.spectral_rolloff(y=audio,
                                               sr=22050,
                                               S=None,
                                               n_fft=2048,
                                               hop_length=512,
                                               freq=None,
                                               roll_percent=0.85)
    return np.nanmean(np.trim_zeros(rolloff[0]))


def getfeatures(audio):
    featurearray = np.zeros(len(featuresnames()))
    featurearray[0] = compute_flatness(audio)
    featurearray[1] = compute_rolloff(audio)

    framedata = frameanalisys.getframefeatures(audio)
    featurearray[2] = framedata[0]
    featurearray[3] = framedata[1]
    return featurearray
