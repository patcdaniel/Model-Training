import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import transforms
import zipfile
import pandas as pd


def stratified_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the dataset into train, validation, and test sets while maintaining class proportions.
    """
    species_labels = np.array([dataset[i][1].item() for i in range(len(dataset))])  # Extract species labels

    # Stratified split
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=42)
    train_idx, temp_idx = next(strat_split.split(np.zeros(len(species_labels)), species_labels))

    # Further split temp set into validation and test
    temp_labels = species_labels[temp_idx]
    strat_split_val_test = StratifiedShuffleSplit(n_splits=1, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
    val_idx, test_idx = next(strat_split_val_test.split(np.zeros(len(temp_labels)), temp_labels))

    # Map indices back to the original dataset
    val_idx, test_idx = temp_idx[val_idx], temp_idx[test_idx]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    return train_dataset, val_dataset, test_dataset

def transform_image(image):
    """ Preprocesses the input image for the model. Mean and STD are based on a subset of images from April IFCB 161. Maybe be different for other instruments (104)."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5516488552093506, 0.5516488552093506, 0.5516488552093506], [0.09317322075366974, 0.09317322075366974, 0.09317322075366974])
        ])
    return transform(image)


def transform_zip_images(image):
    """ Image transformation for images from a zip. Different from .h5 images, which are pre-resized"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.552, 0.552, 0.552],
                             std=[0.0932, 0.0932, 0.0932]),
    ])
    
    return transform(image)


def get_manifest(zip_path):
    """
    Reads the manifest.csv file from the given zip file and returns the DataFrame.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('manifest.csv') as f:
            manifest = pd.read_csv(f)
    return manifest