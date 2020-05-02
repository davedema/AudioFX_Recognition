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
    return 1


def framefeats():
    return 1


def datapathtest():
    return 'C:/Users/jacop/PycharmProjects/CMLShomework/executables/modules/smalldbtest/Guitar/{}'


def datapathtrain():
    return 'C:/Users/jacop/PycharmProjects/CMLShomework/executables/modules/smalldbtrain/Guitar/{}'


def do_plot():
    return True
