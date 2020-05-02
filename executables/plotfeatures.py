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


# param dict_train_features feature dictionary
def plotfeats(dict_train_features):
    for c in arange(len(classes())):
        for i in arange(len(featuresnames())):
            feature = dict_train_features[c].transpose()[i]

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 1, 1)
            plt.stem(feature, use_line_collection=True)
            plt.ylabel(featuresnames()[i] + ' coefficients')
            plt.title(featuresnames()[i] + ' coefficients {}'.format(classes()[c]))
            plt.grid(True)
            plt.show()


if __name__ == "__main__":  # executable from terminal
    from executables.modules import dataloader

    dict_tr = [(dataloader.dict_train_features(c) - dataloader.featmin()) /
               (dataloader.featmax() - dataloader.featmin()) for c in classes()]
    plotfeats(dict_tr)
