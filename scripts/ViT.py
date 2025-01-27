import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel


class SimilarityVisualizer:
    def __init__(self, output_dir: str = "data/results"):
        """Initialize the visualizer with an output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_results(
        self,
        query_image_path: str,
        similar_images: List[Tuple[str, float]],
        session_id: str,
        experiment_title: str,
    ) -> str:
        """Create a visual grid comparing query image with similar images."""
        # Create session directory
        session_dir = self.output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Prepare the plot
        n_similar = len(similar_images)
        plt.figure(figsize=(15, 3 + (n_similar // 3) * 4))
        plt.suptitle(f"{experiment_title}\n{session_id}", fontsize=12, y=1.02)

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
        viz_name = f"{Path(query_image_path).stem}_results.png"
        viz_path = session_dir / viz_name
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
        """Initialize the image similarity search system with a ViT model."""
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(
            model_name, attn_implementation="sdpa", torch_dtype=torch.float16
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore
        self.catalog_embeddings = {}
        self.visualizer = SimilarityVisualizer()
        self.experiment_name = experiment_name
        self.session_id = (
            f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
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
        """Generate embedding for a single image with augmentations."""
        augmentations = [
            image,
            image.rotate(90),
            image.rotate(180),
            image.rotate(270),
            image.transpose(Image.FLIP_LEFT_RIGHT),
            image.transpose(Image.FLIP_TOP_BOTTOM),
        ]

        embeddings = []
        for aug_image in augmentations:
            inputs = self.processor(images=aug_image, return_tensors="pt")  # type: ignore
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings.append(outputs.pooler_output.cpu().numpy()[0])

        # Average the embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding

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
                    query_image_path, results, self.session_id, self.experiment_name
                )
                # print(f"\nVisualization saved to: {viz_path}")
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

    def hits_at_k(
        self,
        test_dir: str,
        labels_csv: str,
        top_ks: Optional[List[int]] = None,
    ) -> Tuple[Dict[int, int], int, Dict[int, float]]:
        """
        Evaluate hits@k metrics for given test images and labels.
        Returns:
            Tuple containing:
                - Dictionary of hit counts for each k
                - Total number of queries processed
                - Dictionary of hit rates for each k
        """
        if top_ks is None:
            top_ks = [1, 3, 5, 10]
        max_k = max(top_ks)

        # Load test labels with extension-agnostic matching
        test_labels = {}
        try:
            with open(labels_csv, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) < 2:
                        continue  # Skip invalid rows

                    # Remove extensions from both test image and correct images
                    test_image = Path(row[0].strip()).stem
                    correct_images = [
                        Path(img.strip()).stem for img in row[1].split(",")
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
                similar_images = self.find_similar_images(
                    str(test_file), top_k=max_k, save_results=True
                )
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


def main():
    try:
        # Initialize the search system with experiment name
        experiment_name = "bg_removed_evaluation"
        searcher = ImageSimilaritySearch(experiment_name=experiment_name)
        visualizer = SimilarityVisualizer()

        # Preprocess images with background removal

        # Process catalog images
        # remove_backgrounds(
        #     input_folder="data/DAM",
        #     output_folder="data/DAM_bg_removed",
        # )

        # # Process test images
        # remove_backgrounds(
        #     input_folder="data/test_image_headmind",
        #     output_folder="data/test_bg_removed",
        # )

        # Build catalog with bg-removed images
        searcher.build_catalog("data/DAM")

        # Evaluate using bg-removed test images
        hit_counts, total_queries, hit_rates = searcher.hits_at_k(
            test_dir="data/test_bg_removed", labels_csv="data/labels.csv"
        )

        # Print results
        print("\nEvaluation Results:")
        for k in sorted(hit_rates.keys()):
            print(f"Hit@{k}: {hit_rates[k]:.4f} ({hit_counts[k]}/{total_queries})")

        test_images = os.listdir("data/test_bg_removed")
        for test_image in test_images[:5]:  # Visualize first 5 test images
            query_image_path = os.path.join("data/test_bg_removed", test_image)
            results = searcher.find_similar_images(
                query_image_path, top_k=5, save_results=False
            )
            visualizer.visualize_results(
                query_image_path, results, searcher.session_id, experiment_name
            )
        print(f"\nAll results saved to: data/results/{searcher.session_id}")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
