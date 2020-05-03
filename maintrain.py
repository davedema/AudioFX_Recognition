from environment.modules import trainingloop, featureselection, savetraindata
import plotfeatures
from environment.modules.analysislab import user_interface
import numpy as np


def train():
    path = user_interface.datapathtrain()
    dict_train_features = trainingloop.getdicttrainfeatures(path)  # compute train features
    X_train = [dict_train_features[c] for c in user_interface.classes()]
    y_train = [np.ones(X_train[i].shape[0], ) * i for i in np.arange(len(user_interface.classes()))]  # keys
    feat_max = np.max(np.concatenate((X_train[0], X_train[1], X_train[2]), axis=0), axis=0)
    feat_min = np.min(np.concatenate((X_train[0], X_train[1], X_train[2]), axis=0), axis=0)
    X_train_normalized = [(X_train[c] - feat_min) / (feat_max - feat_min)
                          for c in np.arange(len(user_interface.classes()))]  # normalized matrix
    if user_interface.do_plot():
        plotfeatures.plotfeats(X_train_normalized)  # plot train features
    X_train_mc_normalized = np.concatenate((X_train_normalized[0], X_train_normalized[1], X_train_normalized[2]),
                                           axis=0)
    y_train_mc = np.concatenate((y_train[0], y_train[1], y_train[2]), axis=0)
    featurelst = featureselection.getfeaturelist(X_train_mc_normalized, y_train_mc)  # feature selection
    savetraindata.savedata(dict_train_features, featurelst, feat_max, feat_min)  # save data
    print('feature matrix:')
    print(user_interface.featuresnames())
    print(X_train_normalized)
    print('\n\nSelected matrix:')
    print([user_interface.featuresnames()[i] for i in featurelst['selectedcolumns']])
    print(featurelst['featurematrix'])
    print('\nfeature scores:')
    print(user_interface.featuresnames())
    print(featurelst['scores'])
    print('\nselected features')
    print([user_interface.featuresnames()[i] for i in featurelst['selectedcolumns']])
    return True


if __name__ == "__main__":  # executable from terminal
    train()
