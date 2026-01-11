import torch
from extract_features import SELDFeatureExtractor
from parameters import params

def main():
    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features()
    feature_extractor.preprocess_features()


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    main()