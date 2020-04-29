import matplotlib.pyplot as plt
from modules import glob


def plotfeats(dict_train_features):
    # guardate la barra colorata a destra, vedrete che il range è molto

    for c in glob.classes():
        tremolo = dict_train_features[c].transpose()[0]

        # Visualization
        fig = plt.figure(figsize=(18, 4))
        plt.subplot(1, 1, 1)
        plt.stem(tremolo)
        plt.ylabel('tremolo coefficients')
        plt.title('TREMOLO COEFFICIENTS {}'.format(c))
        plt.grid(True)
        plt.show()

    for c in glob.classes():
        flatness = dict_train_features[c].transpose()[1]
        # Visualization
        fig = plt.figure(figsize=(18, 4))
        plt.subplot(1, 1, 1)
        plt.stem(flatness)
        plt.ylabel('flatness')
        plt.title('FLATNESS {}'.format(c))
        plt.grid(True)
        plt.show()

    for c in glob.classes():
        rolloff = dict_train_features[c].transpose()[2]
        # Visualization
        fig = plt.figure(figsize=(18, 4))
        plt.subplot(1, 1, 1)
        plt.stem(rolloff)
        plt.ylabel('rolloff coefficients')
        plt.title('ROLLOFF COEFFICIENTS {}'.format(c))
        plt.grid(True)
        plt.show()

    for c in glob.classes():
        centroid = dict_train_features[c].transpose()[3]
        # Visualization
        fig = plt.figure(figsize=(18, 4))
        plt.subplot(1, 1, 1)
        plt.stem(centroid)
        plt.ylabel('centroid coefficients')
        plt.title('TREMOLO2 COEFFICIENTS {}'.format(c))
        plt.grid(True)
        plt.show()
