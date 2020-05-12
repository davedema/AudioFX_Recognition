import sklearn.svm
import numpy as np
from sklearn.model_selection import cross_val_score
from environment.modules.analysislab.user_interface import kfold


def getpredictions(X_train_normalized, y_train, X_test_mc_normalized):
    SVM_parameters = {
        'C': 1,
        'kernel': 'rbf',
    }
    clf_01 = sklearn.svm.SVC(**SVM_parameters, probability=True)
    clf_02 = sklearn.svm.SVC(**SVM_parameters, probability=True)
    clf_12 = sklearn.svm.SVC(**SVM_parameters, probability=True)

    X_01 = np.concatenate((X_train_normalized[0], X_train_normalized[1]), axis=0)
    y_01 = np.concatenate((y_train[0], y_train[1]), axis=0)
    X_02 = np.concatenate((X_train_normalized[0], X_train_normalized[2]), axis=0)
    y_02 = np.concatenate((y_train[0], y_train[2]), axis=0)
    X_12 = np.concatenate((X_train_normalized[1], X_train_normalized[2]), axis=0)
    y_12 = np.concatenate((y_train[1], y_train[2]), axis=0)
    clf_01.fit(X_01, y_01)
    clf_02.fit(X_02, y_02)
    clf_12.fit(X_12, y_12)
    print("\nCross validated scores, k = 5: \n")
    scores = ['accuracy', 'precision_micro', 'recall_macro', 'f1_macro', 'roc_auc']
    overall_scores = {'accuracy': [], 'precision_micro': [], 'recall_macro': [], 'f1_macro': [], 'roc_auc': []}
    for s in scores:
        score1 = cross_val_score(clf_01, X_01, y_01, cv=kfold(), scoring=s)
        score2 = cross_val_score(clf_02, X_02, y_02, cv=kfold(), scoring=s)
        score3 = cross_val_score(clf_12, X_12, y_12, cv=kfold(), scoring=s)
        score0 = np.concatenate((score1, score2, score3), axis=0)
        score = np.average(score0)
        print('\n')
        print(s)
        print('NoFX/Distortion')
        print(score1)
        print('NoFX/Tremolo')
        print(score2)
        print('Distortion/Tremolo')
        print(score3)

        overall_scores[s] = score
    print(overall_scores)
    y_test_predicted_01 = clf_01.predict(X_test_mc_normalized).reshape(-1, 1)
    y_test_predicted_02 = clf_02.predict(X_test_mc_normalized).reshape(-1, 1)
    y_test_predicted_12 = clf_12.predict(X_test_mc_normalized).reshape(-1, 1)
    y_test_predicted_mc = np.concatenate((y_test_predicted_01, y_test_predicted_02, y_test_predicted_12), axis=1)
    y_test_predicted_mc = np.array(y_test_predicted_mc, dtype=np.int)
    y_test_predicted_mv = np.zeros((y_test_predicted_mc.shape[0],))
    for i, e in enumerate(y_test_predicted_mc):
        y_test_predicted_mv[i] = np.bincount(e).argmax()
    return y_test_predicted_mv
