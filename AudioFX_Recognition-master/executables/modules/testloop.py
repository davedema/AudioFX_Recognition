import numpy as np
import librosa
import os
from executables.modules.analysislab import features, user_interface


def getdicttestfeatures(path):
    dict_test_features = {'NoFX': [], 'Distortion': [], 'Tremolo': []}

    for c in user_interface.classes():  # loops over classes
        n_features = len(user_interface.featuresnames())
        test_root = path.format(c)
        # test_root = 'Test/{}/'.format(c)
        class_test_files = [f for f in os.listdir(test_root) if f.endswith('.wav')]
        n_test = len(class_test_files)
        test_features = np.zeros((n_test, n_features))

        for index, f in enumerate(class_test_files):  # loops over all the files of the class

            audio, fs = librosa.load(os.path.join(test_root, f), sr=None)
            test_features[index, :] = features.getfeatures(audio)

        dict_test_features[c] = test_features
    return dict_test_features
