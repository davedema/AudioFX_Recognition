from environment.modules.analysislab.user_interface import featuresnames


def print_features(featurelst):
    print('\n\nSelected matrix:')
    print([featuresnames()[i] for i in featurelst['selectedcolumns']])
    print(featurelst['featurematrix'])
    print('\nfeature scores:')
    print(featuresnames())
    print(featurelst['scores'])
    print('\nselected features')
    print([featuresnames()[i] for i in featurelst['selectedcolumns']])


if __name__ == '__main__':
    from environment.modules import featureselection, dataloader
    from environment.modules.analysislab.user_interface import classes
    import numpy as np

    X_train_normalized = [(dataloader.dict_train_features(c) - dataloader.featmin()) /
                          (dataloader.featmax() - dataloader.featmin()) for c in classes()]
    y_train = [np.ones(X_train_normalized[i].shape[0], ) * i for i in np.arange(len(classes()))]
    X_train_mc_normalized = np.concatenate((X_train_normalized[0], X_train_normalized[1], X_train_normalized[2]),
                                           axis=0)
    y_train_mc = np.concatenate((y_train[0], y_train[1], y_train[2]), axis=0)
    featurelst = featureselection.getfeaturelist(X_train_mc_normalized, y_train_mc)
    print_features(featurelst)
