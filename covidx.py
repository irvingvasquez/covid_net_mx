import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from skimage import io, transform

from PIL import Image

class2id = test_count = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}

id2class = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

class COVIDxDataset(Dataset):
    def __init__(self, txt_frame_file, images_path, transform=None):
        """
        Args:
            txt_frame_file (string): Path to the txt files with labels.
            images_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.covidx_frame = pd.read_csv(txt_frame_file, delim_whitespace=True) 
        
        #self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = images_path
        self.transform = transform
        self.class2id = test_count = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
        self.id2class = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}
        
        
    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    
    def __len__(self):
        return len(self.covidx_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.root_dir,
                                self.covidx_frame.iloc[idx, 1])
        #print(img_name)
        
        image = self.pil_loader(image_path)
        #image = io.imread(img_name)
        
        
        label = self.covidx_frame.iloc[idx, 2]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'label': label}

        if self.transform is not None:
            image = self.transform(image)
            
        sample = (image, self.class2id[label])

        return sample
        
        