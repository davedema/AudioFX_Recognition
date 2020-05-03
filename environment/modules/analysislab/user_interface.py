# this variables must be modified manually before code execution

from numpy import floor
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
    return ['Flatness', 'Rolloff', 'Max in waveform', 'Tremolo feature']


def kbest():
    return 3


def framefeats():
    return 2


def datapathtest():
    return 'C:/Users/jacop/PycharmProjects/CMLShomework/environment/modules/analysislab/smalldbtest/{}'


def datapathtrain():
    return 'C:/Users/jacop/PycharmProjects/CMLShomework/environment/modules/analysislab/smalldbtrain/{}'


def do_plot():
    return True
