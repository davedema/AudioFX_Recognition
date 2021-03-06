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
        'Flatness', 'Rolloff', 'Avg. centroid', 'Spectral bandwidth', 'Zero crossing rate', 'Tremolo feature',
        'Tremolo feature2', 'Mfc1', 'Mfc2', 'Mfc3', 'Mfc4',
        'Mfc5', 'Mfc6', 'Mfc7', 'Mfc8', 'Mfc9', 'Mfc10', 'Mfc11', 'Mfc12', 'Mfc13', 'Mfc14', 'Mfc15', 'Mfc16',
        'Mfc17', 'Mfc18', 'Mfc19', 'Mfc20']


def kbest():
    return 5


def framefeats():
    return 2


def do_plot():
    return False


def kfold():
    return 5


def generate_datasets():
    return True


def test_size():
    return 0.4


def amplitude_scale():
    return 1
