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
    return [
        'Flatness', 'Rolloff', 'Tremolo feature', 'Tremolo feature2', 'Avg. centroid', 'Spectral bandwidth',
        'Zero crossing rate', 'Mfc1', 'Mfc2', 'Mfc3', 'Mfc4'
        , 'Mfc5', 'Mfc6', 'Mfc7', 'Mfc8', 'Mfc9', 'Mfc10', 'Mfc11', 'Mfc12', 'Mfc13', 'Mfc14', 'Mfc15', 'Mfc16', 'Mfc17'
        , 'Mfc18', 'Mfc19', 'Mfc20']


def kbest():
    return 4


def framefeats():
    return 2


def datapathtest():
    return 'C:/Users/david/Documents/polimi/MATERIOZZE/primo anno/CMLS/DATABASE/Test/{}'


def datapathtrain():
    return 'C:/Users/david/Documents/polimi/MATERIOZZE/primo anno/CMLS/DATABASE/Train/{}'


def datapath():
    return 'C:/Users/david/Documents/polimi/MATERIOZZE/primo anno/CMLS/DATABASE/Test/{}'


def do_plot():
    return False


def generate_datasets():
    return True


def test_size():
    return 0.4


def kfold():
    return 5
