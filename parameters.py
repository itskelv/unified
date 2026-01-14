params = {
    'root_dir': '../unifieddataset',
    'feat_dir': '../unifieddataset/features',
    'norm_feat_dir': '../unifieddataset/norm_features',
    'label_dir': '../unifieddataset/labels',
    'unique_classes': 13,
    'sampling_rate': 24000,
    'hop_len': 0.02,
    'label_hop_len': 0.1,
    'nb_mels': 64,

    'batch_size': 128,
    'label_sequence_length': 50,
    'model_dir': 'models/',
    'dropout_rate': 0.05,
    'nb_cnn2d_filt': 64,  
    'f_pool_size': [4, 4, 2],

    'nb_heads': 8,
    'nb_self_attn_layers': 2,
    'nb_transformer_layers': 2,

    'nb_rnn_layers': 2,
    'rnn_size': 128,

    'nb_fnn_layers': 1,
    'fnn_size': 128,

    'nb_epochs': 250,
    'lr': 1e-3,
}

params['feature_label_resolution'] = int(params['label_hop_len'] // params['hop_len'])
params['feature_sequence_length'] = params['label_sequence_length'] * params['feature_label_resolution']
params['t_pool_size'] = [params['feature_label_resolution'], 1, 1]
params['patience'] = int(params['nb_epochs'])