from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel


class ProductDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        processor: ViTImageProcessor,
        is_training: bool = True,
    ):
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

        # Contrastive learning augmentations
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                # transforms.RandomGrayscale(p=0.2),
                # transforms.GaussianBlur(3),
                # transforms.RandomSolarize(threshold=0.5, p=0.1),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        transform = self.train_transform if self.is_training else self.val_transform
        view1 = transform(image)
        view2 = transform(image)  # Second augmented view for contrastive learning

        # Process images through ViT processor
        inputs1 = self.processor(images=view1, return_tensors="pt", do_resize=False)
        inputs2 = self.processor(images=view2, return_tensors="pt", do_resize=False)

        return (
            {k: v.squeeze(0) for k, v in inputs1.items()},
            {k: v.squeeze(0) for k, v in inputs2.items()},
            sample["label"],
        )


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Create similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create labels for positive pairs (same image augmentations)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Discard diagonal entries
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask

        # Compute log softmax
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # Compute mean log-likelihood of positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss


class ViTContrastiveRetrieval:
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViTImageProcessor.from_pretrained(model_name)

        # Initialize base ViT model
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.projection = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        ).to(self.device)

        self.catalog_embeddings = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("contrastive_results") / self.session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 2e-5,
        model_save_path: str = "models/vit_contrastive_2.pth",
    ):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters()},
                {"params": self.projection.parameters()},
            ],
            lr=learning_rate,
        )

        criterion = ContrastiveLoss()
        best_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for view1, view2, labels in progress_bar:
                # Move data to device
                view1 = {k: v.to(self.device) for k, v in view1.items()}
                view2 = {k: v.to(self.device) for k, v in view2.items()}
                labels = labels.to(self.device)

                # Combine views and labels
                combined_views = {
                    k: torch.cat([view1[k], view2[k]]) for k in view1.keys()
                }
                combined_labels = torch.cat([labels, labels])

                # Forward pass
                outputs = self.model(**combined_views)
                features = outputs.last_hidden_state[:, 0, :]
                projections = self.projection(features)

                # Compute contrastive loss
                loss = criterion(projections, combined_labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = self._validate(val_loader, criterion)

            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "projection": self.projection.state_dict(),
                    },
                    model_save_path,
                )
                print(f"Saved new best model with validation loss: {avg_val_loss:.4f}")

    def _validate(self, val_loader: DataLoader, criterion) -> float:
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for view1, view2, labels in val_loader:
                view1 = {k: v.to(self.device) for k, v in view1.items()}
                view2 = {k: v.to(self.device) for k, v in view2.items()}
                labels = labels.to(self.device)

                combined_views = {
                    k: torch.cat([view1[k], view2[k]]) for k in view1.keys()
                }
                combined_labels = torch.cat([labels, labels])

                outputs = self.model(**combined_views)
                features = outputs.last_hidden_state[:, 0, :]
                projections = self.projection(features)

                loss = criterion(projections, combined_labels)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def load_weights(self, model_path: str, strict: bool = True):
        """Load pretrained model weights from .pth file

        Args:
            model_path: Path to saved .pth file
            strict: Whether to enforce exact parameter matching
        """
        # Load checkpoint dictionary
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint["model"], strict=strict)

        # Load projection head weights
        self.projection.load_state_dict(checkpoint["projection"], strict=strict)

        print(f"Successfully loaded weights from {model_path}")
        return self

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Get embedding from the last hidden state (without projection)."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()[0]

        return embedding

    def build_catalog(self, catalog_dir: str):
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
        query_image = Image.open(query_image_path).convert("RGB")
        query_embedding = self.get_embedding(query_image)

        similarities = []
        for path, embedding in self.catalog_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((path, float(similarity)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_k]

        self._visualize_results(query_image_path, results)
        return results

    def _visualize_results(self, query_path: str, results: List[Tuple[str, float]]):
        n_similar = len(results)
        fig = plt.figure(figsize=(15, 3 + (n_similar // 3) * 4))

        plt.subplot(1 + n_similar // 3, 3, 1)
        query_img = Image.open(query_path)
        plt.imshow(query_img)
        plt.title("Query Image", pad=10)
        plt.axis("off")

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
    # df = pd.read_csv("data/product_list.csv")
    # categories = sorted(df["Product_BusinessUnitDesc"].unique())

    vit = ViTContrastiveRetrieval()
    processor = vit.processor

    # full_dataset = ProductDataset("data/DAM", "data/product_list.csv", processor)
    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     full_dataset, [train_size, val_size]
    # )

    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4)

    # vit.train(train_loader, val_loader, num_epochs=8)

    # Load pretrained weights
    vit.load_weights("models/vit_contrastive_2.pth")
    vit.build_catalog("data/DAM")

    test_dir = "data/test_image_headmind"
    for test_file in Path(test_dir).glob("*"):
        print(f"\nProcessing test image: {test_file}")
        similar_images = vit.find_similar_images(str(test_file))

        print("Most similar catalog images:")
        for path, similarity in similar_images:
            print(f"Similarity: {similarity:.3f} - Path: {path}")


if __name__ == "__main__":
    main()
