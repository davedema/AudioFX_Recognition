from modules import trainingloop
from modules import plotfeatures
from modules import savetraindata
from modules import testloop
from modules import glob
from modules import dataloader
from modules import featureselection
from modules import supportvectormachines
from modules import confusionmatrix
import numpy as np

# begin of train main
dict_train_features = trainingloop.getdicttrainfetures()  # compute train features
plotfeatures.plotfeats(dict_train_features)  # plot train features
X_train = [dict_train_features[c] for c in glob.classes()] # seems useless just the same as dict_train_features
y_train = [np.ones(X_train[i].shape[0],) * i for i in np.arange(3)]  # dictionaries
feat_max = np.max(np.concatenate((X_train[0], X_train[1], X_train[2]), axis=0), axis=0)
feat_min = np.min(np.concatenate((X_train[0], X_train[1], X_train[2]), axis=0), axis=0)
X_train_normalized = [(X_train[c] - feat_min) / (feat_max - feat_min)
                      for c in np.arange(len(glob.classes()))]  # normalized matrix
X_train_mc_normalized = np.concatenate((X_train_normalized[0], X_train_normalized[1], X_train_normalized[2]), axis=0)
X_train_mc = np.concatenate((X_train[0], X_train[1], X_train[2]), axis=0)
y_train_mc = np.concatenate((y_train[0], y_train[1], y_train[2]), axis=0)
featurelst = featureselection.getfeaturelist(X_train_mc_normalized, y_train_mc)
savetraindata.savedata(dict_train_features, featurelst, feat_max, feat_min)  # save train features and parameters
print('feature matrix:')
print(X_train_normalized)
print('\n\nSelected matrix:')
print(featurelst['featurematrix'])
print('\nselected features as columns of input matrix')
print(featurelst['selectedcolumns'])
print('\nboolean array identifying columns')
print(featurelst['booleandict'])
#  end of train main

# begin of test main
dict_test_features = testloop.getdicttestfeatures()  # compute test features
#dict_train_features = dataloader.Trainselected()
# TODO: Qui dobbiamo selezionare le colonne delle matrici
X_test = [dict_test_features[c] for c in glob.classes()] # same as X_train
y_test = [np.ones(X_test[i].shape[0],) * i for i in np.arange(3)]  # dictionaries
y_test_mc = np.concatenate((y_test[0], y_test[1], y_test[2]), axis=0)
# X_test_normalized = [((X_test[c] - dataloader.featmin()) / (dataloader.featmax() - dataloader.featmin())) for c in np.arange(len(glob.classes()))]  # normalized matrix
X_test_normalized = [((X_test[c] - feat_min) / (feat_max - feat_min))
                     for c in np.arange(len(glob.classes()))]  # normalized matrix
X_test_mc_normalized = np.concatenate((X_test_normalized[0], X_test_normalized[1], X_test_normalized[2]), axis=0)
y_test_predicted_mv = supportvectormachines.getpredictions(X_train_normalized, y_train, X_test_mc_normalized)  # SVM
print('\n\nConfusion matrix:')
confusionmatrix.compute_cm_multiclass(y_test_mc, y_test_predicted_mv)  # print confusion matrix
# end of test main
