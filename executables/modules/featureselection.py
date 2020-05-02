from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from executables.modules.analysislab.user_interface import kbest


def getfeaturelist(x_train_mc, y_train_mc):
    selecta = SelectKBest(mutual_info_regression, k=kbest()).fit(x_train_mc,
                                                           y_train_mc)  # feature selection object declaration
    columns_selected = selecta.get_support(indices=True)  # selected columns of input matrix
    X_new = selecta.fit_transform(x_train_mc, y_train_mc)  # numpy matrix of selected features
    scores_array = selecta.scores_
    featurelist = {
        'featurematrix': X_new, 'selectedcolumns': columns_selected, 'scores': scores_array}
    return featurelist
