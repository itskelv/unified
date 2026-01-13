params = {
    'root_dir': '../unifieddataset',
    'feat_dir': '../unifieddataset/features',
    'norm_feat_dir': '../unifieddataset/norm_features',
    'label_dir': '../unifieddataset/labels',
    'unique_class': 13,
    'sampling_rate': 24000,
    'hop_len': 0.02,
    'label_hop_len': 0.1,
    'nb_mels': 64,

    'batch_size': 128,
    'label_sequence_length': 50
}

params['feature_label_resolution'] = int(params['label_hop_len'] // params['hop_len'])
params['feature_sequence_lenght'] = params['label_sequence_length'] * params['feature_label_resolution']