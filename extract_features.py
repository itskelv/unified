import os
import glob
import torch
import scipy.io.wavfile as wav
import numpy as np
import librosa

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
        self.root_dir = os.path.join(params['root_dir'], os.path.join(params['root_dir'], self._dataset_combination))
        self.feat_dir = params['feat_dir']
        self.desc_dir = os.path.join(self.root_dir, 'metadata_dev')
        self.fs = params['sampling_rate']
        self.hop_len_s = params['hop_len']
        self.hop_len = int(self.fs * self.hop_len_s)
        self.label_hop_len_s = params['label_hop_len']
        self.label_hop_len = int(self.fs * self.label_hop_len_s)
        self.win_len = 2 * self.hop_len
        self.nfft = 2 ** (self.win_len - 1).bit_length()
        self.nb_mel_bins = params['nb_mels']
        self.nb_channels = 4
        self._eps = 1e-8
        self.filewise_frames = {}

        self.nb_mels = params['nb_mels']
        self.mel_wts = librosa.filters.mel(sr=self.fs, n_fft=self.nfft, n_mels=self.nb_mel_bins).T

    def create_folder(self, folder_name):
        if not os.path.exists(folder_name):
            print('{} folder does not exist, creating it.'.format(folder_name))
            os.makedirs(folder_name)

    def load_audio(self, audio_path):
        fs, audio = wav.read(audio_path)
        audio = audio[:, :self.nb_channels] / 32768.0 + self._eps
        return audio, fs

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
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat

    def extract_file_feature(self, arg_in):
        file_cnt, wav_path, feat_path = arg_in
        spect = self.get_spectrogram_for_file(wav_path)

        # extract mel
        mel_spect = self.get_mel_spectrogram(spect)

        feat = None

        # extract intensity vectors
        # foa_iv = self.get_foa_intensity_vectors(spect)
        # feat = np.concatenate((mel_spect, foa_iv), axis=-1)
        feat = np.concatenate(mel_spect, axis=-1)
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
    

if __name__ == '__main__':
    # use this space to test if the SELDFeatureExtractor class works as expected.
    # All the classes will be called from the main.py for actual use.
    from parameters import params
    params['multiACCDOA'] = False
    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features(split='dev')
    feature_extractor.extract_labels(split='dev')