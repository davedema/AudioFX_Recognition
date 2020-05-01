from executables.modules import testloop, confusionmatrix, dataloader, supportvectormachines
from executables.modules import glob
import numpy as np


def test():
    dict_test_features = testloop.getdicttestfeatures()  # compute test features
    # TODO: Qui dobbiamo selezionare le colonne delle matrici
    X_test = [dict_test_features[c] for c in glob.classes()]  # same as X_train
    columns_selected = dataloader.colums_selected()
    X_test_selected = [X_test[i][:, columns_selected] for i in np.arange(len(glob.classes()))]
    y_test = [np.ones(X_test[i].shape[0], ) * i for i in np.arange(3)]  # dictionaries
    y_test_mc = np.concatenate((y_test[0], y_test[1], y_test[2]), axis=0)
    X_test_normalized = [
        ((X_test_selected[c] - dataloader.featmin()[columns_selected]) /
         (dataloader.featmax()[columns_selected] - dataloader.featmin()[columns_selected]))
        for c in np.arange(len(glob.classes()))]  # normalized matrix
    X_train_normalized_loaded = [
        (dataloader.dict_train_features(c)[:, columns_selected] - dataloader.featmin()[columns_selected]) /
        (dataloader.featmax()[columns_selected] - dataloader.featmin()[columns_selected])
        for c in glob.classes()]
    X_train_normalized_loaded_selected = [
        X_train_normalized_loaded[i][:, columns_selected] for i in np.arange(len(glob.classes()))]
    X_test_mc_normalized = np.concatenate((X_test_normalized[0], X_test_normalized[1], X_test_normalized[2]), axis=0)
    y_train_selected = [np.ones(X_train_normalized_loaded_selected[i].shape[0], ) * i for i in np.arange(3)]
    y_test_predicted_mv = supportvectormachines.getpredictions(X_train_normalized_loaded_selected, y_train_selected,
                                                               X_test_mc_normalized)  # SVM
    print('\n\nConfusion matrix:')
    confusionmatrix.compute_cm_multiclass(y_test_mc, y_test_predicted_mv)  # print confusion matrix
    return True


if __name__ == "__main__":  # executable from terminal
    test()
