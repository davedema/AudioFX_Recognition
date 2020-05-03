from environment.modules.analysislab import user_interface
import numpy as np


def savedata(dict_train_features, featurelst, feat_max, feat_min):
    for c in user_interface.classes():
        dict_train_features[c].dump('dict_train_features_' + c + '.dat')
    np.array([feat_max, feat_min]).dump('feat_max_min.dat')
    featurelst['featurematrix'].dump('features_selected.dat')
    featurelst['selectedcolumns'].dump('columns_selected.dat')
    return True
