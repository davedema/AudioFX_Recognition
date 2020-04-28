import numpy as np
import librosa
import os
from modules import glob
from modules import features


def getdicttestfeatures():
    dict_test_features = {'NoFX': [], 'Distortion': [], 'Tremolo': []}

    for c in glob.classes():  # loops over classes
        n_features = 4
        test_root = 'C:/Users/jacop/PycharmProjects/CMLShomework/modules/smalldbtest/Guitar/{}'.format(c)
        # test_root = 'Test/{}/'.format(c)
        class_test_files = [f for f in os.listdir(test_root) if f.endswith('.wav')]
        n_test = len(class_test_files)
        test_features = np.zeros((n_test, n_features))

        for index, f in enumerate(class_test_files):  # loops over all the files of the class

            audio, fs = librosa.load(os.path.join(test_root, f), sr=None)
            test_win_number = int(np.floor((audio.shape[0] - glob.winlength()) / glob.hopsize()))
            test_features_frame = np.zeros((test_win_number, n_features))

            for i in np.arange(test_win_number):
                frame = audio[i * glob.hopsize(): i * glob.hopsize() + glob.winlength()]
                frame_wind = frame * glob.window()
                frame_windo = np.ndarray(shape=len(frame_wind), buffer=frame_wind)
                spec = np.fft.fft(frame_wind)
                speco = np.fft.fft(frame_windo)
                nyquist = int(np.floor(spec.shape[0] / 2))
                spec = spec[0:nyquist]

                test_features_frame[i, 0] = glob.maxrange(frame)  # waveform frame by frame
                # test_features_frame[i, 1] = compute_zcr(frame_wind, Fs)

            max_in_waveform = np.amax(test_features_frame[:, 0])
            test_features_frame[:, 0] = test_features_frame[:, 0] / max_in_waveform  # normalize waveform

            test_features[index, 0] = features.tremolo_feature(test_features_frame[:, 0])
            flatness = features.compute_flatness(audio)
            rolloff = features.compute_rolloff(audio)

            test_features[index, 1] = np.nanmean(np.trim_zeros(flatness[0]))
            test_features[index, 2] = np.nanmean(np.trim_zeros(rolloff[0]))
            test_features[index, 3] = features.tremolo_feature_2(test_features_frame[:, 0], 0.4, 0.7)

        dict_test_features[c] = test_features
    return dict_test_features
