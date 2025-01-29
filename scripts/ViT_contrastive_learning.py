import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_sim = torch.cosine_similarity(anchor, positive)
        neg_sim = torch.cosine_similarity(anchor, negative)
        losses = torch.relu(neg_sim - pos_sim + self.margin)
        return torch.mean(losses)


class TripletDataset(Dataset):
    def __init__(
        self, dam_image_paths: List[str], mmc_to_path: Dict[str, str], processor
    ):
        self.dam_image_paths = dam_image_paths
        self.mmc_list = [Path(p).stem for p in dam_image_paths]
        self.mmc_to_path = mmc_to_path
        self.processor = processor
        self.transform = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.RandomPerspective(distortion_scale=0.2),
            ]
        )

    def __len__(self):
        return len(self.dam_image_paths)

    def __getitem__(self, idx):
        anchor_mmc = self.mmc_list[idx]
        anchor_path = self.mmc_to_path[anchor_mmc]

        # Load and process anchor image with two augmentations
        anchor_img = Image.open(anchor_path).convert("RGB")
        anchor_aug1 = self.transform(anchor_img)
        anchor_aug2 = self.transform(anchor_img)

        # Select negative sample
        negative_mmc = random.choice(
            [mmc for mmc in self.mmc_list if mmc != anchor_mmc]
        )
        negative_img = Image.open(self.mmc_to_path[negative_mmc]).convert("RGB")

        # Process images through ViT processor
        return {
            "anchor": self.processor(images=anchor_aug1, return_tensors="pt")[
                "pixel_values"
            ][0],
            "positive": self.processor(images=anchor_aug2, return_tensors="pt")[
                "pixel_values"
            ][0],
            "negative": self.processor(images=negative_img, return_tensors="pt")[
                "pixel_values"
            ][0],
        }


class SimilarityVisualizer:
    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_results(
        self,
        query_image_path: str,
        similar_images: List[Tuple[str, float]],
        session_id: str,
        experiment_title: str,
    ) -> str:
        session_dir = self.output_dir / session_id
        session_dir.mkdir(exist_ok=True)

        plt.figure(figsize=(15, 5))
        plt.suptitle(f"{experiment_title}\n{session_id}", fontsize=12)

        plt.subplot(1, 5, 1)
        query_img = Image.open(query_image_path)
        plt.imshow(query_img)
        plt.title("Query Image")
        plt.axis("off")

        for i, (img_path, similarity) in enumerate(similar_images[:4], 2):
            plt.subplot(1, 5, i)
            similar_img = Image.open(img_path)
            plt.imshow(similar_img)
            plt.title(f"Similarity: {similarity:.3f}")
            plt.axis("off")

        viz_path = session_dir / f"{Path(query_image_path).stem}_results.png"
        plt.tight_layout()
        plt.savefig(viz_path, bbox_inches="tight")
        plt.close()

        return str(viz_path)


class ImageSimilaritySearch:
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        experiment_name: str = "default_experiment",
    ):
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name, add_pooling_layer=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.catalog_embeddings = {}
        self.visualizer = SimilarityVisualizer()
        self.experiment_name = experiment_name
        self.session_id = (
            f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        (self.visualizer.output_dir / self.session_id).mkdir(exist_ok=True)

    def load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().numpy().flatten()

    def build_catalog(self, catalog_dir: str):
        catalog_path = Path(catalog_dir)
        self.catalog_embeddings = {}

        for img_path in tqdm(list(catalog_path.glob("*.*")), desc="Building catalog"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                image = self.load_image(str(img_path))
                self.catalog_embeddings[str(img_path)] = self.get_embedding(image)
            except Exception as e:
                print(f"Skipping {img_path.name}: {str(e)}")

    def train_contrastive(self, dam_dir: str, epochs: int = 5, batch_size: int = 8):
        dam_path = Path(dam_dir)
        dam_image_paths = [
            str(p)
            for p in dam_path.glob("*.*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        mmc_to_path = {Path(p).stem: p for p in dam_image_paths}

        dataset = TripletDataset(dam_image_paths, mmc_to_path, self.processor)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        criterion = TripletLoss(margin=1.0)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        scaler = (
            torch.amp.GradScaler(device="cuda") if self.device.type == "cuda" else None
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                anchors = batch["anchor"].to(self.device)
                positives = batch["positive"].to(self.device)
                negatives = batch["negative"].to(self.device)

                with torch.amp.autocast("cuda", enabled=scaler is not None):
                    anchor_emb = self.model(anchors).pooler_output
                    positive_emb = self.model(positives).pooler_output
                    negative_emb = self.model(negatives).pooler_output
                    loss = criterion(anchor_emb, positive_emb, negative_emb)

                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

            print(f"Epoch {epoch + 1} Average Loss: {total_loss / len(dataloader):.4f}")

        self.model.eval()

    def find_similar_images(
        self, query_image_path: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        query_image = self.load_image(query_image_path)
        query_embedding = self.get_embedding(query_image)

        similarities = []
        for img_path, emb in self.catalog_embeddings.items():
            similarity = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            similarities.append((img_path, similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    def evaluate_retrieval(
        self,
        test_dir: str,
        labels_csv: str,
        top_ks: List[int] = [1, 3, 5, 10],
    ) -> Tuple[Dict[int, int], int, Dict[int, float]]:
        """
        Evaluate hits@k metrics for given test images and labels.
        Returns:
            Tuple containing:
                - Dictionary of hit counts for each k
                - Total number of queries processed
                - Dictionary of hit rates for each k
        """
        max_k = max(top_ks)

        # Load test labels with extension-agnostic matching using pandas
        test_labels = {}
        try:
            df = pd.read_csv(labels_csv)
            for _, row in df.iterrows():
                if pd.isna(row.iloc[0]) or pd.isna(row.iloc[1]):
                    continue  # Skip invalid rows

                # Remove extensions from both test image and correct images
                test_image = Path(row.iloc[0].strip()).stem
                correct_images = [
                    Path(img.strip()).stem for img in row.iloc[1].split(",")
                ]

                test_labels[test_image] = correct_images
        except Exception as e:
            print(f"Error loading labels from {labels_csv}: {e}")
            return {}, 0, {}

        # Process test images with extension-agnostic matching
        test_path = Path(test_dir)
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        test_files = [
            f for f in test_path.glob("*") if f.suffix.lower() in image_extensions
        ]

        hit_counts = {k: 0 for k in top_ks}
        total_queries = 0

        for test_file in tqdm(test_files, desc="Evaluating hits@k"):
            # Use stem for extension-agnostic matching
            test_stem = test_file.stem

            if test_stem not in test_labels:
                print(f"Skipping unlabeled image: {test_file.name}")
                continue

            correct_images = test_labels[test_stem]

            try:
                # Get similar images (using full filename for search)
                similar_images = self.find_similar_images(str(test_file), top_k=max_k)
            except Exception as e:
                print(f"Skipping {test_file.name} due to error: {e}")
                continue

            # Compare using stems without extensions
            similar_stems = [Path(path).stem for path, _ in similar_images]

            total_queries += 1
            for k in top_ks:
                if any(correct in similar_stems[:k] for correct in correct_images):
                    hit_counts[k] += 1

        # Calculate hit rates
        hit_rates = {
            k: hit_counts[k] / total_queries if total_queries > 0 else 0.0
            for k in top_ks
        }

        return hit_counts, total_queries, hit_rates

    def save_model(self, save_path: str):
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")

    def load_model(self, load_path: str):
        self.model.load_state_dict(
            torch.load(load_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)  # type: ignore
        print(f"Model weights loaded from {load_path}")


def main():
    try:
        searcher = ImageSimilaritySearch(experiment_name="contrastive_vit")
        print("Starting contrastive training...")
        # searcher.train_contrastive(dam_dir="data/DAM", epochs=5)

        # searcher.save_model("models/contrastive_learning.pth")
        # searcher.load_model("models/contrastive_learning.pth")

        print("\nBuilding catalog embeddings...")
        searcher.build_catalog("data/DAM")

        print("\nEvaluating retrieval performance on original test images...")
        hit_counts, total_queries, hit_rates = searcher.evaluate_retrieval(
            test_dir="data/test_image_headmind", labels_csv="data/labels.csv"
        )

        print("\nEvaluation Results:")
        for k in sorted(hit_rates.keys()):
            print(f"Hit@{k}: {hit_rates[k]:.4f} ({hit_counts[k]}/{total_queries})")

        print(
            "\nEvaluating retrieval performance on test images with backgrounds removed..."
        )
        hit_counts, total_queries, hit_rates = searcher.evaluate_retrieval(
            test_dir="data/test_bg_removed", labels_csv="data/labels.csv"
        )

        print("\nEvaluation Results:")
        for k in sorted(hit_rates.keys()):
            print(f"Hit@{k}: {hit_rates[k]:.4f} ({hit_counts[k]}/{total_queries})")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
