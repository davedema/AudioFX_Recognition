import numpy as np
import librosa
import os
from executables.modules import glob
from executables.modules import features


def getdicttrainfetures():
    # train david
    dict_train_features = {'NoFX': [], 'Distortion': [], 'Tremolo': []}
    # root = os.path.dirname('trainingloop.py')
    # fullpath = os.path.join(root, 'smalldbtrain/Guitar/{}/')
    fullpath = 'C:/Users/jacop/PycharmProjects/CMLShomework/executables/modules/smalldbtrain/Guitar/{}'

    # train_root = 'Train/{}/'.format(c)

    for c in glob.classes():  # loops over classes
        n_features = 4
        train_root = fullpath.format(c)
        class_train_files = [f for f in os.listdir(train_root) if f.endswith('.wav')]
        n_train = len(class_train_files)
        train_features = np.zeros((n_train, n_features))

        for index, f in enumerate(class_train_files):  # loops over all the files of the class

            audio, fs = librosa.load(os.path.join(train_root, f), sr=None)
            train_win_number = int(np.floor((audio.shape[0] - glob.winlength()) / glob.hopsize()))
            train_features_frame = np.zeros((train_win_number, n_features))

            for i in np.arange(train_win_number):
                frame = audio[i * glob.hopsize(): i * glob.hopsize() + glob.winlength()]
                frame_wind = frame * glob.window()
                frame_windo = np.ndarray(shape=len(frame_wind), buffer=frame_wind)
                spec = np.fft.fft(frame_wind)
                speco = np.fft.fft(frame_windo)
                nyquist = int(np.floor(spec.shape[0] / 2))
                spec = spec[0:nyquist]

                train_features_frame[i, 0] = glob.maxrange(frame)  # waveform frame by frame
                # train_features_frame[i, 1] = compute_zcr(frame_wind, Fs)

            max_in_waveform = np.amax(train_features_frame[:, 0])
            train_features_frame[:, 0] = train_features_frame[:, 0] / max_in_waveform  # normalize waveform

            # Tremolo Feature
            train_features[index, 0] = features.tremolo_feature(train_features_frame[:, 0])

            flatness = features.compute_flatness(audio)
            rolloff = features.compute_rolloff(audio)
            train_features[index, 1] = np.nanmean(np.trim_zeros(flatness[0]))
            train_features[index, 2] = np.nanmean(np.trim_zeros(rolloff[0]))
            train_features[index, 3] = features.tremolo_feature_2(train_features_frame[:, 0], 0.4, 0.7)

            # train_features[index, 3] = np.nanmean(np.trim_zeros(train_features_frame[:,3]))

        dict_train_features[c] = train_features
    return dict_train_features
