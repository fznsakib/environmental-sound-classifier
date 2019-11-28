import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'LMC':
            # Edit here to load and concatenate the neccessary features to 
            # create the LMC feature
            feature = {'features' : [self.dataset[index]['features']['logmelspec'], 
                                     self.dataset[index]['features']['chroma'],
                                     self.dataset[index]['features']['spectral_contrast'],
                                     self.dataset[index]['features']['Tonnetz']]}
            
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
            print(feature)        
        elif self.mode == 'MC':
            # Edit here to load and concatenate the neccessary features to 
            # create the MC feature
            feature = {'features' : [self.dataset[index]['features']['mfcc'],
                                     self.dataset[index]['features']['chroma'],
                                     self.dataset[index]['features']['spectral_contrast'],
                                     self.dataset[index]['features']['Tonnetz']]}
            
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            # Edit here to load and concatenate the neccessary features to 
            # create the MLMC feature
            feature = {'features': self.dataset[index]['features'] }
           
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
       
        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label, fname

    def __len__(self):
        return len(self.dataset)