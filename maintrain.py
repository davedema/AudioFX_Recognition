import numpy as np
import main_traintest
import plotfeatures
import plotselected
import print_feature_sel
from environment.modules import trainingloop, featureselection, savetraindata, dataloader
from environment.modules.analysislab import user_interface
import pathlib


def train():
    path = pathlib.Path(__file__).parent.absolute()
    classes = user_interface.classes()
    if user_interface.generate_datasets():
        main_traintest.traintest(path)
        dict_train_features = {'NoFX': [], 'Distortion': [], 'Tremolo': []}
        for c in classes:
            dict_train_features[c] = dataloader.dict_train_features(c)
    else:
        dict_train_features = trainingloop.getdicttrainfeatures(path)  # compute train features
    X_train = [dict_train_features[c] for c in classes]
    y_train = [np.ones(X_train[i].shape[0], ) * i for i in np.arange(len(user_interface.classes()))]  # keys
    feat_max = np.max(np.concatenate((X_train[0], X_train[1], X_train[2]), axis=0), axis=0)
    feat_min = np.min(np.concatenate((X_train[0], X_train[1], X_train[2]), axis=0), axis=0)
    X_train_normalized = [(X_train[c] - feat_min) / (feat_max - feat_min)
                          for c in np.arange(len(user_interface.classes()))]  # normalized matrix
    X_train_mc_normalized = np.concatenate((X_train_normalized[0], X_train_normalized[1], X_train_normalized[2]),
                                           axis=0)
    y_train_mc = np.concatenate((y_train[0], y_train[1], y_train[2]), axis=0)

    featurelst = featureselection.getfeaturelist(X_train_mc_normalized, y_train_mc)  # feature selection
    if user_interface.do_plot():
        plotfeatures.plotfeats(X_train_normalized, mask=np.arange(10) + 7)  # plot train features
    else:
        plotselected.plotsel(X_train_normalized, featurelst['selectedcolumns'])
    savetraindata.savedata(dict_train_features, featurelst, feat_max, feat_min)  # save data
    print('feature matrix:')
    print(user_interface.featuresnames())
    print(X_train_normalized)
    print_feature_sel.print_features(featurelst)
    return True


if __name__ == "__main__":  # executable from terminal
    train()
