import numpy as np


def compute_cm_multiclass(gt, predicted):
    classes = np.unique(gt)

    CM = np.zeros((len(classes), len(classes)))

    for i in np.arange(len(classes)):
        pred_class = predicted[gt == i]

        for j in np.arange(len(pred_class)):
            CM[i, int(pred_class[j])] = CM[i, int(pred_class[j])] + 1
    print(CM)