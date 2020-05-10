from executables.modules import trainingloop, savetraindata
from executables.modules.analysislab import user_interface
import numpy as np
from sklearn.model_selection import train_test_split



def traintest():
    path = user_interface.datapath()
    dict_features = trainingloop.getdicttrainfeatures(path)  # compute train features
    classes = user_interface.classes()
    test_size = user_interface.test_size()

    X = [dict_features[c] for c in classes]
    y = [np.ones(X[i].shape[0], ) * i for i in np.arange(len(classes))]  # keys
    y_mc = np.concatenate((y[0], y[1], y[2]), axis=0)
    X_mc = np.concatenate((X[0], X[1], X[2]), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X_mc, y_mc, test_size=test_size )


    dict_train_feats = {'NoFX': [], 'Distortion': [], 'Tremolo': []}
    dict_test_feats = {'NoFX': [], 'Distortion': [], 'Tremolo': []}
    n_features = len(user_interface.featuresnames())

    for c in np.arange(len(classes)):
        condition = np.mod(y_train, 3) == c
        n_train = len(y_train[condition])

        train_feats = np.zeros((n_train, n_features))
        k=0

        for i in np.arange(len(y_train)):
            if y_train[i] == c:
                train_feats[k,:] = X_train[i, :]
                k = k+1
        dict_train_feats[classes[c]] = train_feats

    for c in np.arange(len(classes)):
        condition = np.mod(y_test, 3) == c
        n_test = len(y_test[condition])

        test_feats = np.zeros((n_test, n_features))
        k=0
        for i in np.arange(len(y_test)):
            if y_test[i] == c:
                test_feats[k,:] = X_test[i, :]
                k = k+1
        dict_test_feats[classes[c]] = test_feats


    savetraindata.save_datasets(dict_train_feats, dict_test_feats)

    return dict_train_feats, dict_test_feats


if __name__ == "__main__":  # executable from terminal
    traintest()
