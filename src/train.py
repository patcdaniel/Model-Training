import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import datetime

from scripts.dataloader import H5Dataset
from scripts.utils import transform_image, stratified_split
from scripts.model import SpeciesBehaviorCNN


# ✅ Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")


def train_model(model, train_loader, val_loader, save_dir, epochs=10, lr=0.001):
    """Train the model with optimizations and display per-batch progress."""
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn_species = nn.CrossEntropyLoss()
    loss_fn_behavior = nn.CrossEntropyLoss()

    # ✅ Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"model_{timestamp}.pth")
    log_path = os.path.join(save_dir, f"training_log_{timestamp}.txt")

    with open(log_path, "w") as log_file:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for images, species_labels, behavior_labels, has_behavior in train_loader:
                    optimizer.zero_grad()
                    images, species_labels, behavior_labels = (
                        images.to(device, non_blocking=True),
                        species_labels.to(device, non_blocking=True),
                        behavior_labels.to(device, non_blocking=True)
                    )
                    with torch.cuda.amp.autocast():
                        species_out, behavior_out = model(images)
                        loss_species = loss_fn_species(species_out, species_labels)
                        mask = has_behavior.bool()
                        loss_behavior = loss_fn_behavior(behavior_out[mask], behavior_labels[mask]) if mask.sum() > 0 else 0
                        total_loss = 0.6 * loss_species + (0.4 * loss_behavior if mask.sum() > 0 else 0)

                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += total_loss.item()
                    pbar.update(1)
            
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for images, species_labels, behavior_labels, has_behavior in val_loader:
                    images, species_labels, behavior_labels = (
                        images.to(device, non_blocking=True),
                        species_labels.to(device, non_blocking=True),
                        behavior_labels.to(device, non_blocking=True)
                    )
                    species_out, behavior_out = model(images)
                    loss_species = loss_fn_species(species_out, species_labels)
                    mask = has_behavior.bool()
                    loss_behavior = loss_fn_behavior(behavior_out[mask], behavior_labels[mask]) if mask.sum() > 0 else 0
                    total_loss = 0.6 * loss_species + (0.4 * loss_behavior if mask.sum() > 0 else 0)
                    val_loss += total_loss.item()

            log_file.write(f"Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}\n")
            print(f"✅ Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved at {save_path}")
    print(f"📄 Training log saved at {log_path}")
    
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--save_dir", type=str, default="../outputs", help="Directory to save model and logs")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    dataset = H5Dataset(args.h5_path, transform=transform_image)
    train_dataset, val_dataset, test_dataset = stratified_split(dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = SpeciesBehaviorCNN(num_species_classes=55, num_behavior_classes=2)
    train_model(model, train_loader, val_loader, args.save_dir, epochs=args.epochs)
