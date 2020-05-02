import matplotlib.pyplot as plt
from executables.modules.analysislab.user_interface import featuresnames, classes
from numpy import arange


# param feat feature array, param i column of feature, optional
def plotfeature(feat, i=-1):
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 1, 1)
    plt.stem(feat, use_line_collection=True)
    if i != -1:
        plt.ylabel(featuresnames()[i] + ' coefficients')
        plt.title(featuresnames()[i] + ' coefficients {}'.format(classes()[i]))
    plt.grid(True)
    plt.show()


# one plot per class
def plotfeats_per_class(dict_train_features):
    for c in arange(len(classes())):
        plt.figure(figsize=(10, 10))
        for i in arange(len(featuresnames())):
            plt.subplot(len(featuresnames()), 1, i + 1)
            feature = dict_train_features[c].transpose()[i]
            plt.stem(feature, use_line_collection=True)
            if i == 0:
                plt.title(classes()[c] + ' coefficients')
            plt.ylabel(featuresnames()[i])
            plt.grid(True)
        plt.show()


# one plot per feature
def plotfeats(dict_train_features):
    for c in arange(len(featuresnames())):
        plt.figure(figsize=(10, 10))
        for i in arange(len(classes())):
            plt.subplot(len(classes()), 1, i + 1)
            feature = dict_train_features[i].transpose()[c]
            plt.stem(feature, use_line_collection=True)
            if i == 0:
                plt.title(featuresnames()[c] + ' coefficients')
            plt.ylabel(classes()[i])
            plt.grid(True)
        plt.show()


if __name__ == "__main__":  # executable from terminal
    from executables.modules import dataloader

    dict_tr = [(dataloader.dict_train_features(c) - dataloader.featmin()) /
               (dataloader.featmax() - dataloader.featmin()) for c in classes()]
    plotfeats(dict_tr)
