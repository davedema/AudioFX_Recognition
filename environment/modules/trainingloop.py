import numpy as np
import librosa
import os
from environment.modules.analysislab import features, user_interface


def getdicttrainfeatures(path):
    dict_train_features = {'NoFX': [], 'Distortion': [], 'Tremolo': []}
    fullpath = path + '/environment/databases/train/{}'

    for c in user_interface.classes():  # loops over classes
        n_features = len(user_interface.featuresnames())
        train_root = fullpath.format(c)
        class_train_files = [f for f in os.listdir(train_root) if f.endswith('.wav')]
        n_train = len(class_train_files)
        train_features = np.zeros((n_train, n_features))

        for index, f in enumerate(class_train_files):  # loops over all the files of the class
            audio, fs = librosa.load(os.path.join(train_root, f), sr=None)
            train_features[index, :] = features.getfeatures(audio)

        dict_train_features[c] = train_features
    return dict_train_features
