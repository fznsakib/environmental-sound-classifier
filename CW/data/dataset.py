import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):
        feature = 0

        if self.mode == 'LMC':
            # Edit here to load and concatenate the neccessary features to
            # create the LMC feature
            feature = self.dataset[index]['features']['logmelspec']
            feature = np.concatenate([feature, self.dataset[index]['features']['chroma']])
            feature = np.concatenate([feature, self.dataset[index]['features']['spectral_contrast']])
            feature = np.concatenate([feature, self.dataset[index]['features']['tonnetz']])

        elif self.mode == 'MC':
            # Edit here to load and concatenate the neccessary features to
            # create the MC feature
            feature = self.dataset[index]['features']['mfcc']
            feature = np.concatenate([feature, self.dataset[index]['features']['chroma']])
            feature = np.concatenate([feature, self.dataset[index]['features']['spectral_contrast']])
            feature = np.concatenate([feature, self.dataset[index]['features']['tonnetz']])

        elif self.mode == 'MLMC':
            # Edit here to load and concatenate the neccessary features to
            # create the MLMC feature
            feature = self.dataset[index]['features']['mfcc']
            feature = np.concatenate([feature, self.dataset[index]['features']['logmelspec']])
            feature = np.concatenate([feature, self.dataset[index]['features']['chroma']])
            feature = np.concatenate([feature, self.dataset[index]['features']['spectral_contrast']])
            feature = np.concatenate([feature, self.dataset[index]['features']['tonnetz']])

        ###### Printing out visual images ######
        elif self.mode == 'MFCC':
            feature = self.dataset[index]['features']['mfcc']

        elif self.mode == 'LM':
            feature = self.dataset[index]['features']['logmelspec']

        elif self.mode == 'chroma':
            feature = self.dataset[index]['features']['chroma']

        elif self.mode == 'SC':
            feature = self.dataset[index]['features']['spectral_contrast']

        elif self.mode == 'TN':
            feature = self.dataset[index]['features']['tonnetz']
        ###### END ######

        feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)

        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label, fname

    def __len__(self):
        return len(self.dataset)
