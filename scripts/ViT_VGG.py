import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel


class CategoryClassifier:
    def __init__(
        self,
        num_classes: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model = self._initialize_model(num_classes)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _initialize_model(self, num_classes: int) -> nn.Module:
        model = models.vgg16(weights="VGG16_Weights.DEFAULT")
        for param in list(model.parameters())[:-4]:
            param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model.to(self.device)

    def predict(self, image: Image.Image) -> int:
        self.model.eval()
        with torch.no_grad():
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            output = self.model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            return int(predicted.item())


class CategoryAwareImageSimilarity:
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """Initialize the category-aware image similarity search system."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize ViT model for similarity
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.similarity_model = ViTModel.from_pretrained(
            model_name, attn_implementation="sdpa", torch_dtype=torch.float16
        ).to(self.device)

        # Initialize category classifier
        categories = ["W RTW", "W SLG", "W Bags", "W Shoes", "Watches", "W Accessories"]
        self.category_mapping = {i: cat for i, cat in enumerate(categories)}
        self.classifier = CategoryClassifier(len(categories), self.device)

        # Initialize other attributes
        self.catalog_embeddings = {}
        self.catalog_categories = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("similarity_results") / self.session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_catalog_metadata(self, csv_path: str):
        """Load category information for catalog images."""
        df = pd.read_csv(csv_path)
        self.category_metadata = {
            f"{row['MMC']}.jpeg": row["Product_BusinessUnitDesc"]
            for _, row in df.iterrows()
        }

    def load_image(self, image_path: str) -> Image.Image:
        """Load and convert image to RGB."""
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate embedding for a single image."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.similarity_model(**inputs)

        return outputs.pooler_output.cpu().numpy()[0]

    def build_catalog(self, catalog_dir: str, csv_path: str):
        """Build embeddings and load categories for catalog images."""
        self.load_catalog_metadata(csv_path)
        catalog_path = Path(catalog_dir)

        print("Building catalog embeddings and categories...")
        for img_path in tqdm(list(catalog_path.glob("*.jpeg"))):
            try:
                # Get category from metadata
                category = self.category_metadata.get(img_path.name)
                if category is None:
                    print(f"Warning: No category found for {img_path.name}")
                    continue

                # Generate embedding
                image = self.load_image(str(img_path))
                embedding = self.get_embedding(image)

                # Store embedding and category
                str_path = str(img_path)
                self.catalog_embeddings[str_path] = embedding
                self.catalog_categories[str_path] = category

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    def find_similar_images(
        self, query_image_path: str, top_k: int = 5, save_results: bool = True
    ) -> List[Tuple[str, float]]:
        """Find the most similar images within the same category."""
        # Load and classify query image
        query_image = self.load_image(query_image_path)
        query_category_idx = self.classifier.predict(query_image)
        query_category = self.category_mapping[query_category_idx]

        print(f"Query image category: {query_category}")

        # Get query embedding
        query_embedding = self.get_embedding(query_image)

        # Calculate similarities only for images in the same category
        similarities = []
        for path, embedding in self.catalog_embeddings.items():
            if self.catalog_categories[path] == query_category:
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((path, float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_k]

        if save_results:
            self._save_results(query_image_path, query_category, results)
            self._visualize_results(query_image_path, results)

        return results

    def _save_results(
        self, query_path: str, category: str, results: List[Tuple[str, float]]
    ):
        """Save search results to JSON."""
        results_file = self.output_dir / "results_history.json"

        # Load existing results if any
        if results_file.exists():
            with open(results_file, "r") as f:
                history = json.load(f)
        else:
            history = {}

        # Add new results
        history[str(query_path)] = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "similar_images": [
                {"path": path, "similarity": similarity} for path, similarity in results
            ],
        }

        # Save updated results
        with open(results_file, "w") as f:
            json.dump(history, f, indent=2)

    def _visualize_results(self, query_path: str, results: List[Tuple[str, float]]):
        """Create and save visualization."""
        n_similar = len(results)
        fig = plt.figure(figsize=(15, 3 + (n_similar // 3) * 4))

        # Plot query image
        plt.subplot(1 + n_similar // 3, 3, 1)
        query_img = Image.open(query_path)
        plt.imshow(query_img)
        plt.title("Query Image", pad=10)
        plt.axis("off")

        # Plot similar images
        for idx, (img_path, similarity) in enumerate(results, 1):
            plt.subplot(1 + n_similar // 3, 3, idx + 1)
            similar_img = Image.open(img_path)
            plt.imshow(similar_img)
            plt.title(f"Similarity: {similarity:.3f}", pad=10)
            plt.axis("off")

        # Save visualization
        viz_path = self.output_dir / f"comparison_{Path(query_path).stem}.png"
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
        print(f"\nVisualization saved to: {viz_path}")


def main():
    # Initialize the system
    similarity_search = CategoryAwareImageSimilarity()

    # Load the existing pretrained classifier weights
    classifier_weights_path = "models/best_model.pth"
    try:
        similarity_search.classifier.model.load_state_dict(
            torch.load(
                classifier_weights_path,
                weights_only=True,
                map_location=similarity_search.device,
            )
        )
        print(f"Successfully loaded classifier weights from {classifier_weights_path}")
    except Exception as e:
        print(f"Error loading classifier weights from {classifier_weights_path}: {e}")
        return

    # Build the catalog
    similarity_search.build_catalog(
        catalog_dir="data/DAM", csv_path="data/product_list.csv"
    )

    # Process test images
    test_dir = "data/test_image_headmind"
    for test_file in Path(test_dir).glob("*"):
        print(f"\nProcessing test image: {test_file}")
        similar_images = similarity_search.find_similar_images(str(test_file))

        print("Most similar catalog images:")
        for path, similarity in similar_images:
            print(f"Similarity: {similarity:.3f} - Path: {path}")


if __name__ == "__main__":
    main()
