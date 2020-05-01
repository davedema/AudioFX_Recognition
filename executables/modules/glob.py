# this variables must be modified manually before code execution

from numpy import floor, amax
from scipy import signal


def Fs():
    return 44100


def winlength():
    return int(floor(0.01 * Fs()))


def hopsize():
    return int(floor(0.0075 * Fs()))


def window():
    return signal.get_window(window='hanning', Nx=winlength())


def classes():
    return ['NoFX', 'Distortion', 'Tremolo']


def featuresnames():
    return ['Audio_Waveform', 'Centroid', 'Decrease', 'Low_pass AW', 'Autocorrelation', 'Tremolo feature']


# Necessary to compute Audio Waveform

def maxrange(frame):
    maxPos = amax(frame)
    return maxPos
