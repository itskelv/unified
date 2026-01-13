import os
import numpy as np
import random
from collections import deque
import extract_features


class UnifiedDataGenerator(object):
    def __init__(self, params, split=1, shuffle=True, per_file=False):
        self._per_file = per_file
        self._split = split
        self._batch_size = params['batch_size']
        self._feature_seq_len = params['feature_sequence_length']
        self._label_seq_len = params['label_sequence_length']
        self._shuffle = shuffle        
        self._feat_cls = extract_features.SELDFeatureExtractor(params=params)      
        self._label_dir = self._feat_cls.label_dir
        self._feat_dir = self._feat_cls.norm_feat_dir
        
        self._nb_mel_bins = self._feat_cls.nb_mels
        self._nb_classes = self._feat_cls.nb_unique_classes
        
        self._unified_feat_dim = 7 * self._nb_mel_bins
        self._nb_ch = 4
        self._filenames_list = []
        self._file_formats = {}
        
        self._get_filenames_and_formats()
        
        # ADPIT parameters
        self._num_track_dummy = 6
        self._num_axis = 5
        self._num_class = self._nb_classes
        
        self._calculate_batches()
        
        print(f'\nUnified Data Generator:')
        print(f'  Split: {split}')
        print(f'  Total files: {len(self._filenames_list)}')
        print(f'  Stereo: {sum(1 for f in self._file_formats.values() if f == "stereo")}')
        print(f'  FOA: {sum(1 for f in self._file_formats.values() if f == "foa")}')
        print(f'  Unified feature dim: {self._unified_feat_dim}')
        print(f'  Input shape: ({self._batch_size}, {self._nb_ch}, {self._feature_seq_len}, {self._nb_mel_bins})')
    
    def _get_filenames_and_formats(self):
        """Get files and determine their formats"""
        for filename in os.listdir(self._feat_dir):
            if filename.endswith('.npy'):
                try:
                    fold_part = filename.split('_')[0]
                    fold_num = int(fold_part.replace('fold', ''))
                    
                    if isinstance(self._split, (list, np.ndarray)):
                        if fold_num in self._split:
                            self._filenames_list.append(filename)
                    else:
                        if fold_num == self._split:
                            self._filenames_list.append(filename)
                    
                    if filename in self._filenames_list:
                        feat_path = os.path.join(self._feat_dir, filename)
                        feat = np.load(feat_path, mmap_mode='r')
            
                        if feat.shape[1] == 7 * self._nb_mel_bins:
                            self._file_formats[filename] = 'foa'
                        elif feat.shape[1] == 4 * self._nb_mel_bins:
                            self._file_formats[filename] = 'stereo'
                        else:
                            print(f'WARNING: Unknown feature dimension {feat.shape[1]} for {filename}')
                            self._file_formats[filename] = 'unknown'
                            
                except Exception as e:
                    print(f'Error processing {filename}: {e}')
                    continue
    
    def _create_unified_features(self, features, format_type):
        nb_frames = features.shape[0]
        unified = np.zeros((nb_frames, self._unified_feat_dim))
        
        if format_type == 'foa':
            unified = features.copy()
        elif format_type == 'stereo':
            # add padding: [L, R, 0, 0, ILD, IPD, 0]
            unified[:, :2*self._nb_mel_bins] = features[:, :2*self._nb_mel_bins]
            unified[:, 4*self._nb_mel_bins:6*self._nb_mel_bins] = features[:, 2*self._nb_mel_bins:4*self._nb_mel_bins]
        else:
            print(f"ERROR: Unknown format {format_type}")
        
        return unified
    
    def _calculate_batches(self):
        if self._per_file:
            max_frames = 0
            for filename in self._filenames_list:
                feat_path = os.path.join(self._feat_dir, filename)
                feat = np.load(feat_path, mmap_mode='r')
                max_frames = max(max_frames, feat.shape[0])
            
            self._batch_size = int(np.ceil(max_frames / self._feature_seq_len))
            self._nb_total_batches = len(self._filenames_list)
            print(f"  Per-file mode: batch_size adjusted to {self._batch_size}")
        else:
            total_frames = 0
            for filename in self._filenames_list:
                feat_path = os.path.join(self._feat_dir, filename)
                feat = np.load(feat_path, mmap_mode='r')
                usable_frames = feat.shape[0] - (feat.shape[0] % self._feature_seq_len)
                total_frames += usable_frames
            
            self._nb_total_batches = int(np.floor(total_frames / 
                                                 (self._batch_size * self._feature_seq_len)))
            if self._nb_total_batches == 0:
                print(f"WARNING: Not enough data for even one batch. Total frames: {total_frames}")
                print(f"  Batch size: {self._batch_size}, Feature seq len: {self._feature_seq_len}")
                print(f"  Minimum required: {self._batch_size * self._feature_seq_len}")
                self._nb_total_batches = 1  # Force at least one batch
        
        self._feature_batch_seq_len = self._batch_size * self._feature_seq_len
        self._label_batch_seq_len = self._batch_size * self._label_seq_len
        print(f"  Total batches: {self._nb_total_batches}")
    
    def generate(self):
        """Generate batches"""
        if self._shuffle:
            random.shuffle(self._filenames_list)
        
        circ_buf_feat = deque()
        circ_buf_label = deque()
        
        file_idx = 0
        batch_idx = 0
        
        while batch_idx < self._nb_total_batches:
            while (len(circ_buf_feat) < self._feature_batch_seq_len or
                   len(circ_buf_label) < self._label_batch_seq_len):
                
                if file_idx >= len(self._filenames_list):
                    if self._per_file:
                        break
                    else:
                        file_idx = 0
                        if self._shuffle:
                            random.shuffle(self._filenames_list)
                        if len(circ_buf_feat) == 0 and len(circ_buf_label) == 0:
                            break
                
                filename = self._filenames_list[file_idx]
                format_type = self._file_formats.get(filename, 'unknown')
                
                try:
                    feat_path = os.path.join(self._feat_dir, filename)
                    features = np.load(feat_path)
                    features = self._create_unified_features(features, format_type)
                    
                    label_path = os.path.join(self._label_dir, filename)
                    labels = np.load(label_path)
                    
                    if not self._per_file:
                        max_label_frames = labels.shape[0] - (labels.shape[0] % self._label_seq_len)
                        labels = labels[:max_label_frames]
                        
                        nb_label_segments = max_label_frames // self._label_seq_len
                        max_feat_frames = nb_label_segments * self._feature_seq_len
                        features = features[:max_feat_frames]
                    
                    for row in features:
                        circ_buf_feat.append(row)
                    
                    for row in labels:
                        circ_buf_label.append(row)
                    
                    file_idx += 1
                    
                except Exception as e:
                    print(f'Error loading {filename}: {e}')
                    file_idx += 1
                    continue
            
            if (len(circ_buf_feat) < self._feature_batch_seq_len or
                len(circ_buf_label) < self._label_batch_seq_len):
                break
            
            batch_feat = np.zeros((self._feature_batch_seq_len, self._unified_feat_dim))
            batch_label = np.zeros((self._label_batch_seq_len, 
                                    self._num_track_dummy, 
                                    self._num_axis, 
                                    self._num_class))
            
            for i in range(self._feature_batch_seq_len):
                batch_feat[i] = circ_buf_feat.popleft()
            
            for i in range(self._label_batch_seq_len):
                batch_label[i] = circ_buf_label.popleft()
            
            batch_feat = batch_feat.reshape((self._feature_batch_seq_len // self._feature_seq_len,
                                             self._feature_seq_len,
                                             self._unified_feat_dim))
            batch_feat = batch_feat.reshape((batch_feat.shape[0], batch_feat.shape[1],
                                             self._nb_ch, self._nb_mel_bins))
            batch_feat = np.transpose(batch_feat, (0, 2, 1, 3))
            
            batch_label = self._split_sequences(batch_label, self._label_seq_len)
            
            yield batch_feat, batch_label
            batch_idx += 1
    
    def _split_sequences(self, data, seq_len):
        """Split data into sequences"""
        if len(data.shape) == 2:
            nb_seqs = data.shape[0] // seq_len
            return data[:nb_seqs*seq_len].reshape((nb_seqs, seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            nb_seqs = data.shape[0] // seq_len
            return data[:nb_seqs*seq_len].reshape((nb_seqs, seq_len, data.shape[1], data.shape[2]))
        elif len(data.shape) == 4:
            nb_seqs = data.shape[0] // seq_len
            return data[:nb_seqs*seq_len].reshape((nb_seqs, seq_len, 
                                                   data.shape[1], data.shape[2], data.shape[3]))
    
    def get_data_sizes(self):
        feat_shape = (self._batch_size, self._nb_ch, self._feature_seq_len, self._nb_mel_bins)
        
        label_shape = (self._batch_size, self._label_seq_len, 
                       self._num_track_dummy, self._num_axis, self._num_class)
        
        return feat_shape, label_shape
    
    def get_total_batches_in_data(self):
        """Get total number of batches"""
        return self._nb_total_batches
    
    def get_filelist(self):
        """Get list of files"""
        return self._filenames_list
    
    def write_output_format_file(self, output_file, output_dict):
        _fid = open(output_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in output_dict.keys():
            for _value in output_dict[_frame_ind]:
                # Write Cartesian format output. Since baseline does not estimate track count and distance we use fixed values.
                _fid.write('{},{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3]), float(_value[4])))
                # TODO: What if our system estimates track cound and distence (or only one of them)
        _fid.close()