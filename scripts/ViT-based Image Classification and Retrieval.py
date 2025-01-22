from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor


class ProductDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        processor: ViTImageProcessor,
        is_training: bool = True,
    ):
        """
        Dataset for product images with categories.

        Args:
            image_dir: Directory containing product images
            csv_path: Path to CSV file with image-category mappings
            processor: ViT image processor
            is_training: Whether this is for training or evaluation
        """
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.is_training = is_training

        # Load and process category data
        df = pd.read_csv(csv_path)
        self.categories = sorted(df["Product_BusinessUnitDesc"].unique())
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        # Create image-label pairs
        self.samples = []
        for _, row in df.iterrows():
            img_path = self.image_dir / f"{row['MMC']}.jpeg"
            if img_path.exists():
                self.samples.append(
                    {
                        "image_path": str(img_path),
                        "category": row["Product_BusinessUnitDesc"],
                        "label": self.category_to_idx[row["Product_BusinessUnitDesc"]],
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs, sample["label"]


class ViTClassifierRetrieval:
    def __init__(
        self, num_classes: int, model_name: str = "google/vit-base-patch16-224"
    ):
        """
        Initialize ViT model for both classification and retrieval.

        Args:
            num_classes: Number of product categories
            model_name: Pretrained ViT model name
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViTImageProcessor.from_pretrained(model_name)

        # Initialize model for classification
        self.model = ViTForImageClassification.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        ).to(self.device)  # type: ignore

        self.catalog_embeddings = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("similarity_results") / self.session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 2e-5,
        model_save_path: str = "models/vit_classifier.pth",
    ):
        """Train the ViT model for classification."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2
        )
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for batch, labels in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = labels.to(self.device)

                outputs = self.model(**batch, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            train_loss = train_loss / len(train_loader)

            # Validation phase
            val_loss, val_accuracy = self._validate(val_loader)
            scheduler.step(val_loss)

            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Saved new best model with validation loss: {val_loss:.4f}")

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch, labels in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = labels.to(self.device)

                outputs = self.model(**batch, labels=labels)
                total_loss += outputs.loss.item()

                predictions = outputs.logits.argmax(-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(val_loader), correct / total

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Get embedding from the last hidden state."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use the [CLS] token embedding from the last layer
            embedding = outputs.hidden_states[-1][:, 0].cpu().numpy()[0]

        return embedding

    def build_catalog(self, catalog_dir: str):
        """Build embeddings for catalog images."""
        catalog_path = Path(catalog_dir)
        image_files = list(catalog_path.glob("*.jpeg"))

        print(f"Building catalog with {len(image_files)} images...")
        for img_path in tqdm(image_files):
            try:
                image = Image.open(img_path).convert("RGB")
                embedding = self.get_embedding(image)
                self.catalog_embeddings[str(img_path)] = embedding
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    def find_similar_images(
        self,
        query_image_path: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find similar images using embedding similarity."""
        query_image = Image.open(query_image_path).convert("RGB")
        query_embedding = self.get_embedding(query_image)

        # Calculate similarities
        similarities = []
        for path, embedding in self.catalog_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((path, float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_k]

        # Visualize results
        self._visualize_results(query_image_path, results)

        return results

    def _visualize_results(self, query_path: str, results: List[Tuple[str, float]]):
        """Visualize query image and similar images."""
        n_similar = len(results)
        fig = plt.figure(figsize=(15, 3 + (n_similar // 3) * 4))

        # Query image
        plt.subplot(1 + n_similar // 3, 3, 1)
        query_img = Image.open(query_path)
        plt.imshow(query_img)
        plt.title("Query Image", pad=10)
        plt.axis("off")

        # Similar images
        for idx, (img_path, similarity) in enumerate(results, 1):
            plt.subplot(1 + n_similar // 3, 3, idx + 1)
            similar_img = Image.open(img_path)
            plt.imshow(similar_img)
            plt.title(f"Similarity: {similarity:.3f}", pad=10)
            plt.axis("off")

        viz_path = self.output_dir / f"comparison_{Path(query_path).stem}.png"
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
        print(f"\nVisualization saved to: {viz_path}")


def main():
    # Load category data
    df = pd.read_csv("data/product_list.csv")
    categories = sorted(df["Product_BusinessUnitDesc"].unique())
    num_classes = len(categories)

    # Initialize model
    vit = ViTClassifierRetrieval(num_classes=num_classes)

    # Create datasets and dataloaders
    processor = vit.processor
    full_dataset = ProductDataset("data/DAM", "data/product_list.csv", processor)

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Train the model
    vit.train(train_loader, val_loader)

    # Build catalog embeddings
    vit.build_catalog("data/DAM")

    # Process test images
    test_dir = "data/test_image_headmind"
    for test_file in Path(test_dir).glob("*"):
        print(f"\nProcessing test image: {test_file}")
        similar_images = vit.find_similar_images(str(test_file))

        print("Most similar catalog images:")
        for path, similarity in similar_images:
            print(f"Similarity: {similarity:.3f} - Path: {path}")


if __name__ == "__main__":
    main()
