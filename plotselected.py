import plotfeatures


def plotsel(features, colums_selected):
    plotfeatures.plotfeats(features, colums_selected)


if __name__ == "__main__":  # executable from terminal
    from environment.modules import dataloader
    from environment.modules.analysislab.user_interface import classes

    dict_tr = [(dataloader.dict_train_features(c) - dataloader.featmin()) /
               (dataloader.featmax() - dataloader.featmin()) for c in classes()]
    plotsel(dict_tr, dataloader.colums_selected())
