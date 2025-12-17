import os
import glob
import torch
import scipy.io.wavfile as wav
import numpy as np
import librosa
from sklearn import preprocessing
import wave
import contextlib

class SELDFeatureExtractor():
    def __init__(self, params):
        """
        Initializes the SELDFeatureExtractor with the provided parameters.
        Args:
            params (dict): A dictionary containing various parameters for audio/video feature extraction among others.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.params = params
        self._dataset_combination = '{}_{}'.format('foa', 'dev')
        self.root_dir = os.path.join(params['root_dir'], self._dataset_combination)
        self.feat_dir = params['feat_dir']
        self.label_dir = params['label_dir']
        self.desc_dir = os.path.join(self.root_dir, 'metadata_dev')
        self.norm_feat_dir = params['norm_feat_dir']
        self.fs = params['sampling_rate']
        self.hop_len_s = params['hop_len']
        self.hop_len = int(self.fs * self.hop_len_s)
        self.label_hop_len_s = params['label_hop_len']
        self.label_hop_len = int(self.fs * self.label_hop_len_s)
        self.win_len = 2 * self.hop_len
        self.nfft = 2 ** (self.win_len - 1).bit_length()
        self.nb_mel_bins = params['nb_mels']
        self.nb_channels = 4 # limit channels up to 4
        self.nb_unique_classes = params['unique_class']
        self.eps = 1e-8
        self.mel_wts = librosa.filters.mel(sr=self.fs, n_fft=self.nfft, n_mels=self.nb_mel_bins).T
        self.filewise_frames = {}

        self.nb_mels = params['nb_mels']
        self.mel_wts = librosa.filters.mel(sr=self.fs, n_fft=self.nfft, n_mels=self.nb_mel_bins).T

    def create_folder(self, folder_name):
        if not os.path.exists(folder_name):
            print('{} folder does not exist, creating it.'.format(folder_name))
            os.makedirs(folder_name)

    def get_frame_stats(self):
        if len(self.filewise_frames) != 0:
            return

        print('Computing frame stats:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self.root_dir, self.desc_dir, self.feat_dir))
        for sub_folder in os.listdir(self.root_dir):
            loc_aud_folder = os.path.join(self.root_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_aud_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                with contextlib.closing(wave.open(os.path.join(loc_aud_folder, wav_filename), 'r')) as f:
                    audio_len = f.getnframes()
                nb_feat_frames = int(audio_len / float(self.hop_len))
                nb_label_frames = int(audio_len / float(self.label_hop_len))
                self.filewise_frames[file_name.split('.')[0]] = [nb_feat_frames, nb_label_frames]
        return

    def load_audio(self, audio_path):
        fs, audio = wav.read(audio_path)
        audio = audio / 32768.0 + self.eps

        if audio.shape < 4:  # stereo
            L = audio[:, 0]
            R = audio[:, 1]

            W = (L + R) / np.sqrt(2) # mimic omnidirectional
            X = (L - R) / np.sqrt(2) # mimic x axis
            Y = np.zeros_like(W) # fake channel
            Z = np.zeros_like(W) # fake channel

            audio = np.stack([W, X, Y, Z], axis=1)

        else:  # FOA
            audio = audio[:, :4]

        return audio, fs

    def load_output_format_file(self, _output_format_file, cm2m=False):  # TODO: Reconsider cm2m conversion
        """
        Loads DCASE output format csv file and returns it in dictionary format

        :param _output_format_file: DCASE output format CSV
        :return: _output_dict: dictionary
        """
        _output_dict = {}
        _fid = open(_output_format_file, 'r')
        # next(_fid)
        _words = []     # For empty files
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            if len(_words) == 4:  # frame, class idx,  polar coordinates(2) # no distance data, for example in eval pred
                _output_dict[_frame_ind].append([int(_words[1]), 0, float(_words[2]), float(_words[3])])
            if len(_words) == 5:  # frame, class idx, source_id, polar coordinates(2) # no distance data, for example in synthetic data fold 1 and 2
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
            if len(_words) == 6: # frame, class idx, source_id, polar coordinates(2), distance
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])/100 if cm2m else float(_words[5])])
            elif len(_words) == 7: # frame, class idx, source_id, cartesian coordinates(3), distance
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5]), float(_words[6])/100 if cm2m else float(_words[6])])
        _fid.close()
        if len(_words) == 7:
            _output_dict = self.convert_output_format_cartesian_to_polar(_output_dict)
        return _output_dict
    
    def convert_output_format_cartesian_to_polar(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]

                    # in degrees
                    azimuth = np.arctan2(y, x) * 180 / np.pi
                    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                    r = np.sqrt(x**2 + y**2 + z**2)
                    out_dict[frame_cnt].append(tmp_val[0:2] + [azimuth, elevation] + tmp_val[5:])
        return out_dict
    
    def convert_output_format_polar_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    ele_rad = tmp_val[3]*np.pi/180.
                    azi_rad = tmp_val[2]*np.pi/180.

                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append(tmp_val[0:2] + [x, y, z] + tmp_val[4:])
        return out_dict
    
    def get_adpit_labels_for_file(self, _desc_file, _nb_label_frames):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels
        for multi-ACCDOA with Auxiliary Duplicating Permutation Invariant Training (ADPIT)

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 6, 4(=act+XYZ), max_classes]
        """

        se_label = np.zeros((_nb_label_frames, 6, self.nb_unique_classes))  # [nb_frames, 6, max_classes]
        x_label = np.zeros((_nb_label_frames, 6, self.nb_unique_classes))
        y_label = np.zeros((_nb_label_frames, 6, self.nb_unique_classes))
        z_label = np.zeros((_nb_label_frames, 6, self.nb_unique_classes))
        dist_label = np.zeros((_nb_label_frames, 6, self.nb_unique_classes))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:
                active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
                active_event_list_per_class = []
                for i, active_event in enumerate(active_event_list):
                    active_event_list_per_class.append(active_event)
                    if i == len(active_event_list) - 1:  # if the last
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                            dist_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[5]/100.
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            dist_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[5]/100.
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                            dist_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[5]/100.
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            dist_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[5]/100.
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            dist_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[5]/100.
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                            dist_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[5]/100.

                    elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                            dist_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[5]/100.
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            dist_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[5]/100.
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                            dist_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[5]/100.
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            dist_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[5]/100.
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            dist_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[5]/100.
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                            dist_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[5]/100.
                        active_event_list_per_class = []

        label_mat = np.stack((se_label, x_label, y_label, z_label, dist_label), axis=2)  # [nb_frames, 6, 5(=act+XYZ+dist), max_classes]
        return label_mat


    def spectrogram(self, audio_input, nb_frames):
        nb_ch = audio_input.shape[1]
        spectra = []
        for ch_cnt in range(nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self.nfft, hop_length=self.hop_len,
                                        win_length=self.win_len, window='hann')
            spectra.append(stft_ch[:, :nb_frames])
        return np.array(spectra).T

    def get_spectrogram_for_file(self, audio_filename):
        audio_in, fs = self.load_audio(audio_filename)

        nb_feat_frames = int(len(audio_in) / float(self.hop_len))
        nb_label_frames = int(len(audio_in) / float(self.label_hop_len))
        self.filewise_frames[os.path.basename(audio_filename).split('.')[0]] = [nb_feat_frames, nb_label_frames]

        audio_spec = self.spectrogram(audio_in, nb_feat_frames)
        return audio_spec

    def get_mel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self.nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self.mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)) # shape (T, 4, F)

        return mel_feat

    def get_intensity_vectors(self, linear_spectra):
        W = linear_spectra[:, :, 0]
        I = np.real(np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:])
        E = self.eps + (np.abs(W)**2 + ((np.abs(linear_spectra[:, :, 1:])**2).sum(-1)) / 3.0)

        I_norm = I / E[:, :, np.newaxis]
        I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0, 2, 1)), self.mel_wts), (0, 2, 1))
        iv = I_norm_mel.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], self.nb_mel_bins * 3))
        if np.isnan(iv).any():
            print('Feature extraction is generating nan outputs')
            exit()
        return iv

    def extract_file_feature(self, arg_in):
        file_cnt, wav_path, feat_path = arg_in
        spect = self.get_spectrogram_for_file(wav_path)

        # extract mel
        mel_spect = self.get_mel_spectrogram(spect)

        feat = None

        # extract intensity vectors
        iv = self.get_intensity_vectors(spect)
        feat = np.concatenate((mel_spect, iv), axis=-1)

        if feat is not None:
            print('{}: {}, {}'.format(file_cnt, os.path.basename(wav_path), feat.shape))
            np.save(feat_path, feat)

    def extract_features(self):
        # setting up folders
        self.create_folder(self.feat_dir)
        # extraction starts
        print('Extracting spectrogram:')
        print('\t\troot_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(self.root_dir, self.desc_dir, self.feat_dir))

        arg_list = []
        for sub_folder in os.listdir(self.root_dir):
            loc_aud_folder = os.path.join(self.root_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_aud_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                wav_path = os.path.join(loc_aud_folder, wav_filename)
                feat_path = os.path.join(self.feat_dir, '{}.npy'.format(wav_filename.split('.')[0]))
                self.extract_file_feature((file_cnt, wav_path, feat_path))
                arg_list.append((file_cnt, wav_path, feat_path))

    def preprocess_features(self):
        self.create_folder(self.norm_feat_dir)

        # pre-processing starts
        print('Estimating weights for normalizing feature files:')
        print('\t\tfeat_dir: {}'.format(self.feat_dir))

        spec_scaler = preprocessing.StandardScaler()
        for file_cnt, file_name in enumerate(os.listdir(self.feat_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self.feat_dir, file_name))
            spec_scaler.partial_fit(feat_file)
            del feat_file

        print('Normalizing feature files:')
        print('\t\tfeat_dir_norm {}'.format(self.norm_feat_dir))
        for file_cnt, file_name in enumerate(os.listdir(self.feat_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self.feat_dir, file_name))
            feat_file = spec_scaler.transform(feat_file)
            np.save(
                os.path.join(self.norm_feat_dir, file_name),
                feat_file
            )
            del feat_file

        print('normalized files written to {}'.format(self.norm_feat_dir))

    def extract_labels(self):
        self.get_frame_stats()

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self.root_dir, self.desc_dir, self.label_dir))
        self.create_folder(self.label_dir)
        for sub_folder in os.listdir(self.desc_dir):
            loc_desc_folder = os.path.join(self.desc_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_desc_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                nb_label_frames = self.filewise_frames[file_name.split('.')[0]][1]
                desc_file_polar = self.load_output_format_file(os.path.join(loc_desc_folder, file_name))
                desc_file = self.convert_output_format_polar_to_cartesian(desc_file_polar)
                label_mat = self.get_adpit_labels_for_file(desc_file, nb_label_frames)
                print('{}: {}, {}'.format(file_cnt, file_name, label_mat.shape))
                np.save(os.path.join(self.label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)
    

if __name__ == '__main__':
    # use this space to test if the SELDFeatureExtractor class works as expected.
    # All the classes will be called from the main.py for actual use.
    from parameters import params
    params['multiACCDOA'] = False
    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features()
    feature_extractor.preprocess_features()
    feature_extractor.extract_labels()