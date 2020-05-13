from environment.modules import testloop, confusionmatrix, dataloader, supportvectormachines, metrics
from environment.modules.analysislab import user_interface
import numpy as np
import pathlib


def test():
    # begin compute and select features
    path = pathlib.Path(__file__).parent.absolute()
    classes = user_interface.classes()
    if user_interface.generate_datasets():
        dict_test_features = {'NoFX': [], 'Distortion': [], 'Tremolo': []}
        for c in classes:
            dict_test_features[c] = dataloader.dict_test_feats(c)
    else:
        dict_test_features = testloop.getdicttestfeatures(path)  # test features
    X_test = [dict_test_features[c] for c in user_interface.classes()]
    columns_selected = dataloader.columns_selected()  # positions of selected features
    X_test_selected = [X_test[i][:, columns_selected] for i in np.arange(len(user_interface.classes()))]  # selection
    y_test = [np.ones(X_test[i].shape[0], ) * i for i in np.arange(len(user_interface.classes()))]  # keys
    y_test_mc = np.concatenate((y_test[0], y_test[1], y_test[2]), axis=0)
    X_test_normalized = [
        ((X_test_selected[c] - dataloader.featmin()[columns_selected]) /
         (dataloader.featmax()[columns_selected] - dataloader.featmin()[columns_selected]))
        for c in np.arange(len(user_interface.classes()))]  # normalized matrix
    X_train_normalized_loaded = [
        (dataloader.dict_train_features(c) - dataloader.featmin()) /
        (dataloader.featmax() - dataloader.featmin())
        for c in user_interface.classes()]  # train features
    X_train_normalized_loaded_selected = [
        X_train_normalized_loaded[i][:, columns_selected]
        for i in np.arange(len(user_interface.classes()))]  # selection
    X_test_mc_normalized = np.concatenate((X_test_normalized[0], X_test_normalized[1], X_test_normalized[2]), axis=0)
    y_train_selected = [
        np.ones(X_train_normalized_loaded_selected[i].shape[0], ) * i
        for i in np.arange(len(user_interface.classes()))]
    # end compute and select features
    y_test_predicted_mv = supportvectormachines.getpredictions(X_train_normalized_loaded_selected, y_train_selected,
                                                               X_test_mc_normalized)  # SVM
    print('\n\nMetrics:')
    metrics.get_metrics(dict_test_features)
    print('\n\nConfusion matrix:')
    confusionmatrix.compute_cm_multiclass(y_test_mc, y_test_predicted_mv)  # print confusion matrix
    return True


if __name__ == "__main__":  # executable from terminal
    test()
