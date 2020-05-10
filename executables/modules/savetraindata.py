from executables.modules.analysislab import user_interface
import numpy as np



def savedata(dict_train_features, featurelst, feat_max, feat_min):
    datasets = user_interface.generate_datasets()
    classes = user_interface.classes()
    if datasets:
      j = 0
    else:
        for c in classes:
            dict_train_features[c].dump('dict_train_features_' + c + '.dat')
    np.array([feat_max, feat_min]).dump('feat_max_min.dat')
    featurelst['featurematrix'].dump('features_selected.dat')
    featurelst['selectedcolumns'].dump('columns_selected.dat')
    return True

def save_datasets(dict_train_feats, dict_test_feats):
    for c in user_interface.classes():
        dict_train_feats[c].dump('dict_train_feats_' + c + '.dat')

    for c in user_interface.classes():
        dict_test_feats[c].dump('dict_test_feats_' + c + '.dat')
    return True

