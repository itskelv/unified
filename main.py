import torch

from parameters import params
from extract_features import SELDFeatureExtractor

def main():
    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features(split='dev')
    # feature_extractor.extract_labels(split='dev')

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    main()