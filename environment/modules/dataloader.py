from numpy import load as loaddat


def Trainselected():
    return loaddat('features_selected.dat', allow_pickle=True)


def featmax():
    return loaddat('feat_max_min.dat', allow_pickle=True)[0]


def featmin():
    return loaddat('feat_max_min.dat', allow_pickle=True)[1]


def dict_train_features(c):
    return loaddat('dict_train_features_' + c + '.dat', allow_pickle=True)


def colums_selected():
    return loaddat('columns_selected.dat', allow_pickle=True)
