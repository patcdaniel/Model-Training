import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import h5py
import numpy as np
from PIL import Image

class H5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        
        # Open HDF5 file
        with h5py.File(self.h5_path, 'r') as h5_file:
            self.dataset_size = h5_file['images'].shape[0]
            self.has_behavior_label = np.array(h5_file['has_behavior_label'])  # Binary flag for species with behavior classification

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as h5_file:
            image_data = h5_file['images'][idx]
            image = Image.fromarray(image_data)
            
            species_label = h5_file['species_labels'][idx]
            has_behavior = self.has_behavior_label[idx]  # Check if species has a behavior classification
            behavior_label = h5_file['behavior_labels'][idx] if has_behavior else -1  # -1 for N/A

        if self.transform:
            image = self.transform(image)

        return image, species_label, behavior_label, has_behavior

class SpeciesBehaviorCNN(nn.Module):
    def __init__(self, num_species_classes, num_behavior_classes=2):
        super(SpeciesBehaviorCNN, self).__init__()
        
        # Load pretrained Xception model
        self.backbone = models.xception(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the final FC layer
        
        # Define classifiers
        self.species_classifier = nn.Linear(2048, num_species_classes)
        self.behavior_classifier = nn.Linear(2048, num_behavior_classes)  # Binary classifier for behavior
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Predict species and optional behavior
        species_out = self.species_classifier(features)
        behavior_out = self.behavior_classifier(features)
        
        return species_out, behavior_out

def save_model(model, path="species_behavior_cnn.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path="species_behavior_cnn.pth", num_species_classes=55, num_behavior_classes=2):
    model = SpeciesBehaviorCNN(num_species_classes, num_behavior_classes)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and Save Species Behavior CNN on a SLURM Cluster")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default='species_behavior_cnn.pth', help='Path to save the model')
    parser.add_argument('--h5_path', type=str, required=True, help='Path to training images HDF5 file')
    parser.add_argument('--train_split', type=float, default=0.7, help='Training data split fraction')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation data split fraction')
    parser.add_argument('--subset_classes', type=int, nargs='+', help='Specify a subset of species class labels to use for testing')

    args = parser.parse_args()
    
    # Load dataset and apply subset filter if needed
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = H5Dataset(args.h5_path, transform=transform, subset_classes=args.subset_classes)

    # Adjust num_species_classes dynamically
    unique_species_classes = np.unique([dataset.species_labels[i] for i in range(len(dataset))])
    num_species_classes = len(unique_species_classes)
    num_behavior_classes = 2  # Binary classification for behavior (if applicable)

    print(f"✅ Using {num_species_classes} species classes for training.")

    model = SpeciesBehaviorCNN(num_species_classes, num_behavior_classes)

    if args.train:
        # Define optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Split dataset into train, validation, and test
        total_size = len(dataset)
        train_size = int(args.train_split * total_size)
        val_size = int(args.val_split * total_size)
        test_size = total_size - train_size - val_size  # Ensure total size consistency

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        model.train()
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for images, species_labels, behavior_labels, has_behavior in train_loader:
                optimizer.zero_grad()

                # Forward pass
                species_out, behavior_out = model(images)

                # Compute species classification loss
                loss_species = nn.CrossEntropyLoss()(species_out, species_labels)

                # Compute behavior loss only for valid classes
                mask = has_behavior.bool()
                if mask.sum() > 0:
                    loss_behavior = nn.CrossEntropyLoss()(behavior_out[mask], behavior_labels[mask])
                    total_loss = 0.6 * loss_species + 0.4 * loss_behavior
                else:
                    total_loss = loss_species  # Ignore behavior loss if not applicable

                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()

            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        # Save trained model
        save_model(model, args.save_path)