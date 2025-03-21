import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import zipfile
import pandas as pd

class H5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        
        with h5py.File(self.h5_path, 'r') as h5_file:
            self.dataset_size = h5_file['images'].shape[0]
            self.species_labels = np.array(h5_file['species_labels'])
            self.has_behavior_label = np.array(h5_file['has_behavior_label'])
            self.behavior_labels = np.array(h5_file['behavior_labels'])
            
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as h5_file:
            image_data = h5_file['images'][idx]
            image = Image.fromarray(image_data)
            species_label = torch.tensor(self.species_labels[idx], dtype=torch.long)
            has_behavior = self.has_behavior_label[idx]
            behavior_label = torch.tensor(self.behavior_labels[idx] if has_behavior else -1, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        
        return image, species_label, behavior_label, has_behavior
    

class ZipDataset(Dataset):
    """
    Custom Dataset for phytoplankton images stored in a zip file.
    The zip file is expected to contain:
      - Images organized in directories named after the labels.
      - A manifest.csv file with columns 'filename' and 'label'.  
        The 'filename' column should contain the path to the image file within the zip.
    """
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform
        
        # Read manifest.csv from the zip file
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            with z.open('manifest.csv') as f:
                self.manifest = pd.read_csv(f)
        
        # Build a mapping from label strings to integer indices
        labels = self.manifest['label'].unique()
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(labels))}
        self.manifest['target'] = self.manifest['label'].map(self.label_to_index)
        
        # Lazy initialization of the zip file handle (important for DataLoader workers)
        self.zip_file = None
        self.num_classes = len(labels)


    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        # Open the zip file handle if it hasn't been already (each worker gets its own handle)
        if self.zip_file is None:
            self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
            
        row = self.manifest.iloc[idx]
        fname = row['fname']  # Path inside the zip file
        label = row['label']
        target = row['target']
        img_path = f"{label}/{fname}"
        
        
        # Read image from the zip file
        with self.zip_file.open(img_path) as f:
            image = Image.open(f)
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

def custom_collate(batch):
    """Custom collate function to stack images and labels into tensors."""
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    return images, labels



# class ZipDataset(Dataset):
#     """
#     Custom Dataset for phytoplankton images stored in a zip file.
#     The zip file is expected to contain:
#       - Images organized in directories named after the labels.
#       - A manifest.csv file with columns 'filename' and 'label'.  
#         The 'filename' column should contain the path to the image file within the zip.
#     """
#     def __init__(self, zip_path, transform=None):
#         self.zip_path = zip_path
#         self.transform = transform
        
#         # Read manifest.csv from the zip file
#         with ZipFile(self.zip_path, 'r') as z:
#             with z.open('manifest.csv') as f:
#                 self.manifest = pd.read_csv(f)
        
#         # Build a mapping from label strings to integer indices
#         labels = self.manifest['label'].unique()
#         self.label_to_index = {label: idx for idx, label in enumerate(sorted(labels))}
#         self.manifest['target'] = self.manifest['label'].map(self.label_to_index)
#         self.num_classes = len(labels)
        
#         # Lazy initialization of the zip file handle (important for DataLoader workers)
#         self.zip_file = None

#     def __len__(self):
#         return len(self.manifest)

#     def __getitem__(self, idx):
#         # Open the zip file handle if it hasn't been already (each worker gets its own handle)
#         if self.zip_file is None:
#             self.zip_file =  ZipFile(self.zip_path, 'r')
            
#         row = self.manifest.iloc[idx]
#         fname = row['fname']  # Path inside the zip file
#         label = row['label']
#         img_path = f"{label}/{fname}"
        
#         # Read image from the zip file
#         with self.zip_file.open(img_path) as f:
#             image = Image.open(f)
#             image = image.convert('RGB')
        
#         if self.transform:
#             image = self.transform(image)
            
#         return image, label
