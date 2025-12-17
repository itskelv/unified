def get_params():
    # ########### default parameters ##############
    params = dict(
        quick_test=True,  # To do quick test. Trains/test on small subset of dataset, and # of epochs

        finetune_mode=True,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='3_1_dev_split0_multiaccdoa_foa_model.h5',

        # INPUT PATH
        dataset_dir='../unifieddataset/',

        # OUTPUT PATHS
        feat_label_dir='../unifieddataset/seld_feat_label/',

        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',  # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,
    )