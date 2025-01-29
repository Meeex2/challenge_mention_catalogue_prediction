import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# AutoEncoder Model (unchanged)
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, latent_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 16 * 16),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# Custom Dataset with Augmentation
class AugmentedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [
            f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))
        ]

        # Define a comprehensive set of augmentations
        self.augmentations = transforms.Compose(
            [
                # Color jittering (brightness, contrast, saturation, hue)
                transforms.ColorJitter(
                    brightness=0.2,  # Random brightness change
                    contrast=0.2,  # Random contrast change
                    saturation=0.2,  # Random saturation change
                    hue=0.1,  # Random hue change
                ),
                # Random rotations
                transforms.RandomRotation(
                    degrees=(-30, 30),  # Random rotation between -30 and 30 degrees
                    expand=False,
                ),
                # Random horizontal and vertical flips
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                # Affine transformations
                transforms.RandomAffine(
                    degrees=0,  # No additional rotation
                    translate=(0.1, 0.1),  # Random translation
                    scale=(0.9, 1.1),  # Random scaling
                    shear=10,  # Random shearing
                ),
            ]
        )

        self.base_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Apply base transform first
        base_image = self.base_transform(image)

        # Apply augmentations
        aug_image = self.augmentations(image)
        aug_image = self.base_transform(aug_image)

        return base_image, aug_image, img_name


# Custom collate function to handle multiple outputs
def custom_collate(batch):
    base_images = [item[0] for item in batch]
    aug_images = [item[1] for item in batch]
    filenames = [item[2] for item in batch]

    return (torch.stack(base_images), torch.stack(aug_images), filenames)


# Modified Training Loop with Augmentation
def train_autoencoder(num_epochs=500):
    # Load Augmented DAM dataset
    dam_dataset = AugmentedDataset("data/DAM")
    train_size = int(0.8 * len(dam_dataset))
    val_size = len(dam_dataset) - train_size
    train_dataset, val_dataset = random_split(dam_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate,
    )

    # Initialize model, loss, optimizer
    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        train_loss = 0.0

        for base_inputs, aug_inputs, _ in train_loader:
            # Combine base and augmented inputs for better training
            inputs = torch.cat([base_inputs, aug_inputs], dim=0).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset) * 2  # type: ignore

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for base_inputs, aug_inputs, _ in val_loader:
                inputs = torch.cat([base_inputs, aug_inputs], dim=0).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset) * 2  # type: ignore

        tqdm.write(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_autoencoder.pth")

    return model


model = AutoEncoder()

# train_autoencoder(num_epochs=500)
# Load best model
model.load_state_dict(torch.load("best_autoencoder.pth"))
model.to(device)
encoder = model.encoder


# Custom Dataset to load images with filenames
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


# Transformations
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
dam_dataset = CustomDataset("data/DAM", transform=transform)
# Generate DAM embeddings
dam_loader = DataLoader(dam_dataset, batch_size=32, shuffle=False)
dam_embeddings = []
dam_filenames = []
with torch.no_grad():
    for inputs, names in tqdm(
        dam_loader, desc="Generating DAM embeddings", unit="batch"
    ):
        inputs = inputs.to(device)
        emb = encoder(inputs).cpu()
        dam_embeddings.append(emb)
        dam_filenames.extend(names)
dam_embeddings = torch.cat(dam_embeddings, dim=0)

# Generate test embeddings
test_dataset = CustomDataset("data/test_bg_removed", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_embeddings = []
test_filenames = []
with torch.no_grad():
    for inputs, names in tqdm(
        test_loader, desc="Generating test embeddings", unit="batch"
    ):
        inputs = inputs.to(device)
        emb = encoder(inputs).cpu()
        test_embeddings.append(emb)
        test_filenames.extend(names)
test_embeddings = torch.cat(test_embeddings, dim=0)

# Load test labels (no need for os.path.basename since CSV contains filenames)
test_labels = pd.read_csv("data/labels.csv")
test_to_dam = dict(
    zip(test_labels["test_image_name"], test_labels["label_image_name"])
)  # Use actual column name

# Create DAM filename to index mapping
dam_filename_to_index = {name: idx for idx, name in enumerate(dam_filenames)}

# Calculate hits@1,3,5,10
hits = {1: 0, 3: 0, 5: 0, 10: 0}
total = 0

for test_emb, test_file in zip(test_embeddings, test_filenames):
    if test_file not in test_to_dam:
        continue
    true_dam_file = test_to_dam[test_file]
    if true_dam_file not in dam_filename_to_index:
        continue

    true_idx = dam_filename_to_index[true_dam_file]
    distances = torch.norm(dam_embeddings - test_emb, dim=1)

    # Get top 10 nearest neighbors
    _, topk_indices = torch.topk(distances, k=10, largest=False)

    # Check hits for different k values
    for k in [1, 3, 5, 10]:
        if true_idx in topk_indices[:k]:
            hits[k] += 1

    total += 1

# Calculate and print metrics
print("Classification Performance:")
for k in [1, 3, 5, 10]:
    hit_rate = hits[k] / total
    print(f"Hits@{k}: {hit_rate * 100:.2f}%")
