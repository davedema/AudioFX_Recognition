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
    return 1


def datapathtest():
    return 'C:/Users/david/Documents/polimi/MATERIOZZE/primo anno/CMLS/DATABASE/Test/{}'


def datapathtrain():
    return 'C:/Users/david/Documents/polimi/MATERIOZZE/primo anno/CMLS/DATABASE/Train/{}'

def datapath():
    return 'C:/Users/david/Documents/polimi/MATERIOZZE/primo anno/CMLS/DATABASE/Train/{}'

def do_plot():
    return False

def generate_datasets():
    return True

def test_size():
    return 0.3