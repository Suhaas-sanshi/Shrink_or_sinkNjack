import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Import the models we built in model.py
from model import TinyEncoder, SSLPretrainModel, FinalClassifierModel

# DETERMINISTIC SETUP (as mentioned in teh rule bool ) 
def set_seed(seed=42):
    #Ensures completely reproducible results even for the judges
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- DATA PREPARATION ---
class TwoCropTransform:
    """Generates two augmented views of the same image for Contrastive Learning."""
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

def get_dataloaders(data_dir='./data', debug_mode=False):#debug mode (TRUE) ->500 image , 2 epochs FALSE -> 100,000 images and many epochs ; 
    # 1. SSL Transforms (Aggressive distortions)
    ssl_transform = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))
    ])
    
    # 2. Fine-tuning Transforms (Standard augmentations)
    train_transform = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))
    ])

    # Load Datasets (Will auto-download to 'data_dir' if not present)
    unlabeled_ds = torchvision.datasets.STL10(root=data_dir, split='unlabeled', download=True, transform=TwoCropTransform(ssl_transform))
    labeled_ds = torchvision.datasets.STL10(root=data_dir, split='train', download=True, transform=train_transform)

    # Truncate dataset for fast local CPU testing
    if debug_mode:
        print("DEBUG MODE: Using only 500 samples for fast local testing on CPU.")
        unlabeled_ds = Subset(unlabeled_ds, range(500))
        labeled_ds = Subset(labeled_ds, range(500))

    ssl_loader = DataLoader(unlabeled_ds, batch_size=64 if debug_mode else 256, shuffle=True, drop_last=True)
    train_loader = DataLoader(labeled_ds, batch_size=32 if debug_mode else 64, shuffle=True)

    return ssl_loader, train_loader

# --- PHASE 1: SSL LOSS AND TRAINING ---
def contrastive_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    embeddings = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    sim_matrix.masked_fill_(mask, -9e15)
    positives = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(z1.device)
    return F.cross_entropy(sim_matrix, positives)

def train_ssl(model, dataloader, device, epochs=10):
    print("\n--- Starting Phase 1: Self-Supervised Pre-training ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in dataloader:
            view1, view2 = images[0].to(device), images[1].to(device)
            optimizer.zero_grad()
            z1, z2 = model(view1), model(view2)
            loss = contrastive_loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"SSL Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
    return model

# --- PHASE 2: CLASSIFIER TRAINING ---
def train_classifier(model, dataloader, device, epochs=20):
    print("\n--- Starting Phase 2: Supervised Fine-Tuning ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        acc = 100. * correct / total
        print(f"Fine-Tune Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}, Acc: {acc:.2f}%")
    return model

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    DEBUG_MODE = False # <--- SET THIS TO FALSE WHEN TRAINING ON A CLOUD GPU!

    ssl_loader, train_loader = get_dataloaders(data_dir='./data', debug_mode=DEBUG_MODE)

    # Phase 1: Pre-train on unlabeled data
    base_encoder = TinyEncoder()
    ssl_model = SSLPretrainModel(base_encoder).to(device)
    ssl_model = train_ssl(ssl_model, ssl_loader, device, epochs=2 if DEBUG_MODE else 50)

    # Phase 2: Transfer the encoder weights and fine-tune on labeled data
    final_model = FinalClassifierModel(ssl_model.encoder, num_classes=10).to(device)
    final_model = train_classifier(final_model, train_loader, device, epochs=2 if DEBUG_MODE else 50)

    # Save Final Model and Check File Size
    save_path = "final_model.pth"
    torch.save(final_model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Final Model Size: {os.path.getsize(save_path) / 1024:.2f} KB")