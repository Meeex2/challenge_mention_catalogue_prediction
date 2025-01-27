import random
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


class SimilarityDataset(Dataset):
    def __init__(self, csv_file: str, images_dir: str, labels_dir: str):
        """
        Dataset for training with triplets (anchor, positive, negative).

        Args:
            csv_file: Path to CSV with matching pairs
            images_dir: Directory containing anchor images
            labels_dir: Directory containing matching (positive) images
        """
        self.annotations = pd.read_csv(csv_file)
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        # Standard ImageNet normalization
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        )

        self.all_image_names = self.annotations["label_image_name"].tolist()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get anchor and positive image paths
        anchor_name = self.images_dir / str(self.annotations.iloc[idx, 0])
        positive_name = self.labels_dir / str(self.annotations.iloc[idx, 1])

        # Load images
        anchor = Image.open(str(anchor_name).replace(".jpg", ".png")).convert("RGB")
        positive = Image.open(str(positive_name).replace(".jpg", ".png")).convert("RGB")

        # Get random negative
        negative_name = self._get_random_negative(positive_name)
        negative = Image.open(negative_name).convert("RGB")

        # Apply augmentation
        anchor = self.augment(anchor)
        positive = self.augment(positive)
        negative = self.augment(negative)

        # Apply normalization
        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)

        return anchor, positive, negative

    def _get_random_negative(self, positive_name):
        while True:
            neg_name = self.labels_dir / random.choice(self.all_image_names)
            if neg_name != positive_name:
                return neg_name


class ResNetSimilarity(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        """
        ResNet50-based similarity model with custom embedding head.

        Args:
            embedding_dim: Final embedding dimension (default: 256)
        """
        super().__init__()

        # Load pretrained ResNet
        resnet = models.resnet50(pretrained=True)

        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Add custom mapping network
        self.embedding_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.embedding_head(features)
        return nn.functional.normalize(embeddings, p=2, dim=1)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, margin: float = 0.3):
        """
        Temperature-scaled contrastive loss with margin.

        Args:
            temperature: Scaling factor for similarities
            margin: Minimum difference between positive and negative similarities
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature

        # Compute loss with margin
        loss = torch.mean(torch.relu(neg_sim - pos_sim + self.margin))
        return loss


class SimilarityTrainer:
    def __init__(self, model_save_dir: str = "models"):
        """
        Trainer for the ResNet similarity model.

        Args:
            model_save_dir: Directory to save model checkpoints
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_dataset: SimilarityDataset,
        val_dataset: Optional[SimilarityDataset] = None,
        batch_size: int = 32,
        num_epochs: int = 30,
        learning_rate: float = 0.0001,
    ):
        # Initialize model and move to device
        model = ResNetSimilarity().to(self.device)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )

        # Initialize loss and optimizer
        criterion = ContrastiveLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        # Training loop
        best_loss = float("inf")
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_idx, (anchor, positive, negative) in enumerate(
                tqdm(train_loader)
            ):
                # Move to device
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                # Forward pass
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)

                # Compute loss
                loss = criterion(anchor_emb, positive_emb, negative_emb)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if (batch_idx + 1) % 50 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                    )

            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            if val_loader:
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for anchor, positive, negative in val_loader:
                        anchor = anchor.to(self.device)
                        positive = positive.to(self.device)
                        negative = negative.to(self.device)

                        anchor_emb = model(anchor)
                        positive_emb = model(positive)
                        negative_emb = model(negative)

                        loss = criterion(anchor_emb, positive_emb, negative_emb)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                scheduler.step(avg_val_loss)

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                )

                # Save best model
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(
                        model.state_dict(), self.model_save_dir / "best_model.pth"
                    )
            else:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}"
                )
                if avg_train_loss < best_loss:
                    best_loss = avg_train_loss
                    torch.save(
                        model.state_dict(), self.model_save_dir / "best_model.pth"
                    )

        return model


class ImageSimilaritySearch:
    def __init__(self, model_path: str, embedding_dim: int = 256):
        """
        Image similarity search using trained ResNet model.

        Args:
            model_path: Path to trained model weights
            embedding_dim: Embedding dimension matching the trained model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNetSimilarity(embedding_dim=embedding_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.catalog_embeddings = {}

    def build_catalog(self, catalog_dir: str, batch_size: int = 32):
        """Build embeddings for all images in catalog."""
        catalog_path = Path(catalog_dir)
        image_files = list(catalog_path.glob("**/*.jpg")) + list(
            catalog_path.glob("**/*.png")
        )

        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i : i + batch_size]
            batch_images = []

            for img_path in batch_files:
                try:
                    image = Image.open(img_path).convert("RGB")
                    image = self.transform(image)
                    batch_images.append(image)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)

                with torch.no_grad():
                    embeddings = self.model(batch_tensor)

                for img_path, embedding in zip(batch_files, embeddings):
                    self.catalog_embeddings[str(img_path)] = embedding.cpu().numpy()

    def find_similar(
        self, query_image_path: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find most similar images to query image."""
        # Load and process query image
        query_image = Image.open(query_image_path).convert("RGB")
        query_tensor = self.transform(query_image).unsqueeze(0).to(self.device)

        # Get query embedding
        with torch.no_grad():
            query_embedding = self.model(query_tensor).cpu().numpy()[0]

        # Calculate similarities
        similarities = []
        for path, embedding in self.catalog_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((path, float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def create_train_val_split(
    csv_path: str, train_ratio: float = 0.66
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/val split from original CSV file."""
    df = pd.read_csv(csv_path)

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate split index
    split_idx = int(len(df) * train_ratio)

    # Split into train and validation
    train_df = df[:split_idx]
    val_df = df[split_idx:]

    return train_df, val_df


def save_split_csvs(
    train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: str = "data"
):
    """Save train and validation splits to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train_labels.csv", index=False)
    val_df.to_csv(output_dir / "val_labels.csv", index=False)


def visualize_triplets(dataset: SimilarityDataset, num_examples: int = 3):
    """
    Visualize random triplets (anchor, positive, negative) from the dataset.

    Args:
        dataset: SimilarityDataset instance
        num_examples: Number of triplets to visualize
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 5 * num_examples))

    # Function to denormalize images
    def denormalize(tensor):
        # Clone tensor to avoid modifying the original
        tensor = tensor.clone()

        # Denormalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean

        # Clip values to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)

        return tensor

    for i in range(num_examples):
        # Get a random triplet
        anchor, positive, negative = dataset[random.randint(0, len(dataset) - 1)]

        # Create subplots for this triplet
        ax1 = plt.subplot(num_examples, 3, i * 3 + 1)
        ax2 = plt.subplot(num_examples, 3, i * 3 + 2)
        ax3 = plt.subplot(num_examples, 3, i * 3 + 3)

        # Denormalize and convert to numpy arrays
        anchor_img = denormalize(anchor).permute(1, 2, 0).numpy()
        positive_img = denormalize(positive).permute(1, 2, 0).numpy()
        negative_img = denormalize(negative).permute(1, 2, 0).numpy()

        # Plot images
        ax1.imshow(anchor_img)
        ax2.imshow(positive_img)
        ax3.imshow(negative_img)

        # Set titles
        ax1.set_title("Anchor")
        ax2.set_title("Positive")
        ax3.set_title("Negative")

        # Remove axes
        ax1.axis("off")
        ax2.axis("off")
        ax3.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    print("Starting ResNet similarity training pipeline...")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create train/val split
    print("Creating train/val split...")
    train_df, val_df = create_train_val_split("data/test_labels.csv")
    save_split_csvs(train_df, val_df)

    # Initialize trainer
    trainer = SimilarityTrainer(model_save_dir="models/resnet_similarity")

    # Create datasets using the splits
    print("Creating datasets...")
    train_dataset = SimilarityDataset(
        csv_file="data/train_labels.csv",
        images_dir="data/test_bg_removed",  # Your test images directory
        labels_dir="data/DAM",  # Same directory for labels since they're all there
    )

    val_dataset = SimilarityDataset(
        csv_file="data/val_labels.csv",
        images_dir="data/test_bg_removed",
        labels_dir="data/DAM",
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Visualize some training triplets
    print("\nVisualizing training triplets...")
    visualize_triplets(train_dataset, num_examples=3)

    # Train model
    print("\nStarting training...")
    model = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,  # Smaller batch size to handle potential memory constraints
        num_epochs=20,  # Reduced epochs for faster iteration
        learning_rate=0.0001,
    )

    # Initialize search system
    print("\nInitializing search system...")
    searcher = ImageSimilaritySearch(
        model_path="models/resnet_similarity/best_model.pth"
    )

    # Build catalog
    print("Building catalog from DAM images...")
    searcher.build_catalog("data/DAM")

    # Evaluate on test set
    print("\nEvaluating model...")
    test_results = []
    total_hits = {1: 0, 5: 0, 10: 0}
    total_queries = 0

    # Read test labels
    test_df = pd.read_csv("data/test_labels.csv")

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        query_image = f"data/test_bg_removed/{row['label_image_name']}"
        if not Path(query_image).exists():
            continue

        # Get similar images
        similar_images = searcher.find_similar(query_image, top_k=10)

        # Get ground truth matches
        true_matches = set([row["image_name"]])  # Adjust based on your CSV structure

        # Check hits@k
        for k in [1, 5, 10]:
            top_k_results = {Path(path).name for path, _ in similar_images[:k]}
            if any(match in top_k_results for match in true_matches):
                total_hits[k] += 1

        total_queries += 1

    # Print evaluation results
    print("\nFinal Evaluation Results:")
    for k in [1, 5, 10]:
        hit_rate = total_hits[k] / total_queries if total_queries > 0 else 0
        print(f"Hits@{k}: {hit_rate:.4f} ({total_hits[k]}/{total_queries})")

    print("\nTraining and evaluation pipeline completed!")


if __name__ == "__main__":
    main()
