import sklearn.svm
import numpy as np
import itertools

from sklearn.model_selection import cross_val_score
from environment.modules.analysislab.user_interface import kfold,classes

def getpredictions(X_train_normalized, y_train, X_test_mc_normalized):

    scores = ['accuracy','precision_micro','recall_macro','f1_macro','roc_auc']
    overall_scores = {'accuracy': [], 'precision_micro': [], 'recall_macro': [], 'f1_macro': [], 'roc_auc': []}

    couples = list(itertools.combinations(classes(), 2))
    num_couples = len(couples)
    num_test_files = X_test_mc_normalized.shape[0]

    scores_per_couple = np.zeros((num_couples, len(scores)))
    y_test_predicted_mc = []

    for n in np.arange(num_couples):
        class_0 = couples[n][0]
        index_0 = classes().index(class_0)
        class_1 = couples[n][1]
        index_1 = classes().index(class_1)

        SVM_parameters = {
            'C': 1,
            'kernel': 'rbf',
        };

        clf = sklearn.svm.SVC(**SVM_parameters, probability=True)
        X_train_0 = X_train_normalized[index_0]
        X_train_1 = X_train_normalized[index_1]
        y_train_0 = y_train[index_0]
        y_train_1 = y_train[index_1]
        y = np.concatenate((y_train_0, y_train_1))
        X = np.concatenate((X_train_0, X_train_1), axis=0)

        clf.fit(X, y)
        print(class_0,"/",class_1,"\nCross validated scores for kfolds = ", kfold(), ": \n")

        for s in scores:
            score_array = np.around(cross_val_score(clf, X, y, cv=kfold(), scoring=s), decimals=4) *100
            score = np.around(np.average(score_array), decimals=2 )
            print(s,"\t", score_array,"\t\taverage:", score)
            scores_per_couple[n, scores.index(s)] = score
        print("\n")
        y_predicted = clf.predict(X_test_mc_normalized).reshape(-1, 1)
        y_test_predicted_mc = np.append(y_test_predicted_mc, y_predicted)

    for s in scores:
        final_score = np.average(scores_per_couple.transpose()[scores.index(s)])
        overall_scores[s] = np.around(final_score, decimals = 2)

    print("Average of crossvalidated scores considering all binary cases")
    print(overall_scores)

    y_test_predicted_mc = y_test_predicted_mc.reshape(num_couples, num_test_files).transpose()
    y_test_predicted_mc = np.array(y_test_predicted_mc, dtype=np.int)
    y_test_predicted_mv = np.zeros((y_test_predicted_mc.shape[0],))

    for i, e in enumerate(y_test_predicted_mc):
        y_test_predicted_mv[i] = np.bincount(e).argmax()

    return y_test_predicted_mv
