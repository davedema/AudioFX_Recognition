import itertools

import numpy as np
import sklearn.svm

from environment.modules import dataloader
from environment.modules.analysislab.user_interface import classes


def compute_metrics(gt_labels, predicted_labels):
    TP = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 1))
    FP = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 0))
    TN = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 0))
    FN = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 1))
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)

    accuracy = np.around(accuracy, decimals = 4) * 100
    precision = np.around(precision, decimals = 4) * 100
    recall = np.around(recall, decimals = 4) * 100
    F1_score = np.around(F1_score, decimals = 4) * 100
    print("Results : \n accuracy = {} % \n precision = {} %  \n recall = {} % \n F1 score = {} % \n".format(
        accuracy, precision, recall, F1_score))

    return accuracy, precision, recall, F1_score


def compute_overall_metrics(metrics_matrix):
    accuracy = np.average(metrics_matrix.transpose()[0])
    precision = np.average(metrics_matrix.transpose()[1])
    recall = np.average(metrics_matrix.transpose()[2])
    F1_score = np.average(metrics_matrix.transpose()[3])

    accuracy = np.around(accuracy, decimals=4)
    precision = np.around(precision, decimals=4)
    recall = np.around(recall, decimals=4)
    F1_score = np.around(F1_score, decimals=4)

    print("\nOverall Results : \n accuracy = {} % \n precision = {} %  \n recall = {} % \n F1 score = {} % \n".format(
        accuracy, precision, recall, F1_score))


def get_metrics(dict_test_features):
    metrics_matrix = []
    n = len(classes())
    m = 4
    for subset in itertools.combinations(classes(), 2):
        class_0 = subset[0]
        class_1 = subset[1]


        X_train_0 = dataloader.dict_train_features(class_0)
        X_train_1 = dataloader.dict_train_features(class_1)
        X_train = np.concatenate((X_train_0, X_train_1), axis=0)
        X_train = X_train[:, dataloader.columns_selected()]

        y_train_0 = np.zeros((X_train_0.shape[0],))
        y_train_1 = np.ones((X_train_1.shape[0],))
        y_train = np.concatenate((y_train_0, y_train_1), axis=0)

        X_test_0 = dict_test_features[class_0]
        X_test_1 = dict_test_features[class_1]
        X_test = np.concatenate((X_test_0, X_test_1), axis=0)
        X_test = X_test[:, dataloader.columns_selected()]

        y_test_0 = np.zeros((X_test_0.shape[0],))
        y_test_1 = np.ones((X_test_1.shape[0],))
        y_test = np.concatenate((y_test_0, y_test_1), axis=0)

        feat_max = np.max(X_train, axis=0)
        feat_min = np.min(X_train, axis=0)
        X_train_normalized = (X_train - feat_min) / (feat_max - feat_min)
        X_test_normalized = (X_test - feat_min) / (feat_max - feat_min)

        SVM_parameters = {
            'C': 1,
            'kernel': 'rbf',
        }

        clf = sklearn.svm.SVC(**SVM_parameters)

        clf.fit(X_train_normalized, y_train)
        y_test_predicted = clf.predict(X_test_normalized)

        print("{} // {}".format(class_0, class_1))
        metrics_couple = compute_metrics(y_test, y_test_predicted)
        metrics_matrix = np.append(metrics_matrix, metrics_couple)

    metrics_matrix = np.reshape(metrics_matrix, (n, 4))

    compute_overall_metrics(metrics_matrix)
