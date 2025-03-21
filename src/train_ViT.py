import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import timm
import json, os, datetime, logging



from scripts.dataloader import ZipDataset,custom_collate
from scripts.utils import transform_zip_images


def train_one_epoch(model, device, dataloader, criterion, optimizer, logger):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    logger.info(f"Training loss: {epoch_loss:.4f}")
    return epoch_loss

def validate(model, device, dataloader, criterion, logger):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct.double() / len(dataloader.dataset)
    logger.info(f"Validation loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
    return epoch_loss, accuracy.item()

def main():
    parser = argparse.ArgumentParser(description='Train ViT for Phytoplankton Classification using images from a zip file')
    parser.add_argument('--config', type=str, default=None,help='Path to JSON config file with training options')
    parser.add_argument('--data', type=str, default='./data/dataset.zip', help='Path to the zip file containing images and manifest.csv')
    parser.add_argument('--output', type=str, default='./output', help='Path to the zip file containing images and manifest.csv')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    args = parser.parse_args()

    # Override arguments with those from the config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(args, key, value)

    # Set up timestamped directories for outputs and logs
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outputs_dir = os.path.join(args.output, timestamp)
    logs_dir = "../logs"
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"{timestamp}.log")

    # Set up logging
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger()
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info("Starting training script")
    logger.info(f"Using configuration: {args}")

    # Choose device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Load the dataset and create training/validation splits
    dataset = ZipDataset(args.data, transform=transform_zip_images)
    # Load the dataset and create training/validation splits
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,collate_fn=custom_collate)

    # Load the pretrained ViT model from timm and adjust the classifier head for the specified number of classes.
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=dataset.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, device, train_loader, criterion, optimizer, logger)
        val_loss, val_acc = validate(model, device, val_loader, criterion, logger)
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(outputs_dir, 'best_vit_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")


if __name__ == '__main__':
    main()