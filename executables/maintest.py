from modules import testloop
from modules import glob
from modules import dataloader
from modules import supportvectormachines
from modules import confusionmatrix
import numpy as np


dict_test_features = testloop.getdicttestfeatures()  # compute test features
# dict_train_features = dataloader.Trainselected()
# TODO: Qui dobbiamo selezionare le colonne delle matrici
X_test = [dict_test_features[c] for c in glob.classes()]  # same as X_train
y_test = [np.ones(X_test[i].shape[0], ) * i for i in np.arange(3)]  # dictionaries
y_test_mc = np.concatenate((y_test[0], y_test[1], y_test[2]), axis=0)
X_test_normalized = [((X_test[c] - dataloader.featmin()) / (dataloader.featmax() - dataloader.featmin()))
                     for c in np.arange(len(glob.classes()))]  # normalized matrix
X_train_normalized_loaded = [((dataloader.dict_train_features(c) - dataloader.featmin()) /
                              (dataloader.featmax() - dataloader.featmin())) for c in glob.classes()]
X_test_mc_normalized = np.concatenate((X_test_normalized[0], X_test_normalized[1], X_test_normalized[2]), axis=0)
y_test_predicted_mv = supportvectormachines.getpredictions(X_train_normalized_loaded, y_train, X_test_mc_normalized)  # SVM
print('\n\nConfusion matrix:')
confusionmatrix.compute_cm_multiclass(y_test_mc, y_test_predicted_mv)  # print confusion matrix