# this variables must be modified manually before code execution

import numpy as np
import scipy as sp


def Fs():
    return 44100


def winlength():
    return int(np.floor(0.01 * Fs()))


def hopsize():
    return int(np.floor(0.0075 * Fs()))


def window():
    return sp.signal.get_window(window='hanning', Nx=winlength())


def classes():
    return ['NoFX', 'Distortion', 'Tremolo']


def featuresnames():
    return ['Audio_Waveform', 'Centroid', 'Decrease', 'Low_pass AW', 'Autocorrelation', 'Tremolo feature']


# Necessary to compute Audio Waveform

def maxrange(frame):
    maxPos = np.amax(frame)
    return maxPos
