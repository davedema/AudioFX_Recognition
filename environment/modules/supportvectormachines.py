import sklearn.svm
import numpy as np


def getpredictions(X_train_normalized, y_train, X_test_mc_normalized):
    SVM_parameters = {
        'C': 1,
        'kernel': 'rbf',
    }

    clf_01 = sklearn.svm.SVC(**SVM_parameters, probability=True)
    clf_02 = sklearn.svm.SVC(**SVM_parameters, probability=True)
    clf_12 = sklearn.svm.SVC(**SVM_parameters, probability=True)
    clf_01.fit(np.concatenate((X_train_normalized[0], X_train_normalized[1]), axis=0),
               np.concatenate((y_train[0], y_train[1]), axis=0))

    clf_02.fit(np.concatenate((X_train_normalized[0], X_train_normalized[2]), axis=0),
               np.concatenate((y_train[0], y_train[2]), axis=0))

    clf_12.fit(np.concatenate((X_train_normalized[1], X_train_normalized[2]), axis=0),
               np.concatenate((y_train[1], y_train[2]), axis=0))
    y_test_predicted_01 = clf_01.predict(X_test_mc_normalized).reshape(-1, 1)
    y_test_predicted_02 = clf_02.predict(X_test_mc_normalized).reshape(-1, 1)
    y_test_predicted_12 = clf_12.predict(X_test_mc_normalized).reshape(-1, 1)
    y_test_predicted_mc = np.concatenate((y_test_predicted_01, y_test_predicted_02, y_test_predicted_12), axis=1)
    y_test_predicted_mc = np.array(y_test_predicted_mc, dtype=np.int)
    y_test_predicted_mv = np.zeros((y_test_predicted_mc.shape[0],))
    for i, e in enumerate(y_test_predicted_mc):
        y_test_predicted_mv[i] = np.bincount(e).argmax()

    return y_test_predicted_mv
