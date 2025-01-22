import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel


class SimilarityVisualizer:
    def __init__(self, output_dir: str = "data/similarity_results"):
        """Initialize the visualizer with an output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_results(
        self,
        query_image_path: str,
        similar_images: List[Tuple[str, float]],
        session_id: str,
    ) -> str:
        """Create a visual grid comparing query image with similar images."""
        # Create session directory
        session_dir = self.output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Prepare the plot
        n_similar = len(similar_images)
        _ = plt.figure(figsize=(15, 3 + (n_similar // 3) * 4))

        # Plot query image
        plt.subplot(1 + n_similar // 3, 3, 1)
        query_img = Image.open(query_image_path)
        plt.imshow(query_img)
        plt.title("Query Image", pad=10)
        plt.axis("off")

        # Plot similar images
        for idx, (img_path, similarity) in enumerate(similar_images, 1):
            plt.subplot(1 + n_similar // 3, 3, idx + 1)
            similar_img = Image.open(img_path)
            plt.imshow(similar_img)
            plt.title(f"Similarity: {similarity:.3f}", pad=10)
            plt.axis("off")

        # Save the visualization
        viz_path = session_dir / f"comparison_{Path(query_image_path).stem}.png"
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()

        return str(viz_path)


class ImageSimilaritySearch:
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """Initialize the image similarity search system with a ViT model."""
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(
            model_name, attn_implementation="sdpa", torch_dtype=torch.float16
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore
        self.catalog_embeddings = {}
        self.visualizer = SimilarityVisualizer()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_history: Dict[str, List[Dict]] = {}

        # Create session directory immediately
        self.session_dir = self.visualizer.output_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, image_path: str) -> Image.Image:
        """Load and convert image to RGB."""
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

    def get_embedding(self, image: Image.Image) -> torch.Tensor:
        """Generate embedding for a single image."""
        inputs = self.processor(images=image, return_tensors="pt")  # type: ignore
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.pooler_output.cpu().numpy()[0]

    def build_catalog(self, catalog_dir: str):
        """Build embeddings for all images in the catalog directory."""
        catalog_path = Path(catalog_dir)
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

        # Find all image files
        image_files = [
            f for f in catalog_path.rglob("*") if f.suffix.lower() in image_extensions
        ]

        print(f"Building catalog with {len(image_files)} images...")

        # Generate embeddings for each image
        for img_path in tqdm(image_files):
            try:
                image = self.load_image(str(img_path))
                embedding = self.get_embedding(image)
                self.catalog_embeddings[str(img_path)] = embedding
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    def find_similar_images(
        self, query_image_path: str, top_k: int = 20, save_results: bool = True
    ) -> List[Tuple[str, float]]:
        """Find the most similar images to the query image."""
        # Get query image embedding
        query_image = self.load_image(query_image_path)
        query_embedding = self.get_embedding(query_image)

        # Calculate similarities
        similarities = []
        for path, embedding in self.catalog_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((path, float(similarity)))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_k]

        if save_results:
            try:
                # Save results to history
                self.save_results(query_image_path, results)

                # Generate visualization
                viz_path = self.visualizer.visualize_results(
                    query_image_path, results, self.session_id
                )
                print(f"\nVisualization saved to: {viz_path}")
            except Exception as e:
                print(f"Error saving results: {e}")

        return results

    def save_results(self, query_image_path: str, results: List[Tuple[str, float]]):
        """Save search results to history."""
        query_name = Path(query_image_path).name
        if query_name not in self.results_history:
            self.results_history[query_name] = []

        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "similar_images": [
                {"name": Path(path).name, "similarity": similarity}
                for path, similarity in results
            ],
        }
        self.results_history[query_name].append(result_entry)

        # Ensure the directory exists and save results
        results_file = self.session_dir / "results_history.csv"
        try:
            with open(results_file, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["test_image"] + [f"similar_image{i}" for i in range(20)]
                writer.writerow(header)
                for query, entries in self.results_history.items():
                    for entry in entries:
                        row = [query] + [
                            img["name"] for img in entry["similar_images"][:20]
                        ]
                        writer.writerow(row)
        except Exception as e:
            print(f"Error saving results to {results_file}: {e}")
            raise


def main():
    try:
        # Initialize the search system
        searcher = ImageSimilaritySearch()

        # Build the catalog
        catalog_dir = "data/DAM"
        searcher.build_catalog(catalog_dir)

        # Process test images
        test_dir = "data/test_image_headmind"
        test_files = [
            f
            for f in Path(test_dir).glob("*")
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]

        # Find similar images for each test image
        for test_file in test_files:
            print(f"\nProcessing test image: {test_file}")
            similar_images = searcher.find_similar_images(str(test_file))

            print("Most similar catalog images:")
            for path, similarity in similar_images:
                print(f"Similarity: {similarity:.3f} - Path: {path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
