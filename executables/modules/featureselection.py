from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression


def getfeaturelist(X_train_mc, y_train_mc):
    selecta = SelectKBest(mutual_info_regression, k=4).fit(X_train_mc,
                                                           y_train_mc)  # feature selection object declaration
    diction = selecta.get_support()  # X_train_mc[0:-1, dict_trained_features == True] = selected features
    columns_selected = selecta.get_support(indices=True)  # selected columns of input matrix
    X_new = selecta.fit_transform(X_train_mc, y_train_mc)  # numpy matrix of selected features
    featurelist = {'featurematrix': X_new, 'selectedcolumns': columns_selected, 'booleandict':  diction}
    return featurelist
