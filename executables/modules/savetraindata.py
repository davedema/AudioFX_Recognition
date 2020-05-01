from executables.modules import glob
import numpy as np


def savedata(dict_train_features, featurelst, feat_max, feat_min, y_train):
    for c in glob.classes():
        dict_train_features[c].dump('dict_train_features_' + c + '.dat')
    np.array([feat_max, feat_min]).dump('feat_max_min.dat')
    featurelst['featurematrix'].dump('features_selected.dat')
    featurelst['selectedcolumns'].dump('columns_selected.dat')
    np.array(y_train).dump('y_train.dat')
    return True
