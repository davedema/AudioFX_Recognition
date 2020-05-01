import numpy as np


def Trainselected():
    return np.load('features_selected.dat', allow_pickle=True)


def featmax():
    return np.load('feat_max_min.dat', allow_pickle=True)[0]


def featmin():
    return np.load('feat_max_min.dat', allow_pickle=True)[1]


def dict_train_features(c):
    return np.load('dict_train_features_' + c + '.dat', allow_pickle=True)


def colums_selected():
    return np.load('columns_selected.dat', allow_pickle=True)
