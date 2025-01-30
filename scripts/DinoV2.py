import csv
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


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
        #  model_name: str = "google/vit-base-patch16-224-in21k",
        model_name: str = "facebook/dinov2-base",
        experiment_name: str = "default_experiment",
    ):
        """Initialize the image similarity search system with a ViT model."""
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
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

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate embedding for a single image with augmentations and color histograms."""
        augmentations = [
            image,
            image.rotate(90, expand=True),
            image.rotate(180, expand=True),
            image.rotate(270, expand=True),
            image.transpose(Image.FLIP_LEFT_RIGHT),  # type: ignore
            image.transpose(Image.FLIP_TOP_BOTTOM),  # type: ignore
        ]

        embeddings = []
        for aug_image in augmentations:
            inputs = self.processor(images=aug_image, return_tensors="pt")  # type: ignore
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # outputs = self.model(**inputs)
                outputs = self.model(**inputs).last_hidden_state[0]

            # embeddings.append(outputs.pooler_output.cpu().numpy()[0])
            embeddings.append(outputs.cpu().numpy()[0])

        # Average the embeddings
        avg_embedding = np.mean(embeddings, axis=0)

        # Calculate color histograms
        color_histograms = self.calculate_color_histograms(image)

        # Concatenate embeddings with color histograms
        final_embedding = np.concatenate((avg_embedding, color_histograms))

        return final_embedding

    def calculate_color_histograms(self, image: Image.Image) -> np.ndarray:
        """Calculate color histograms for the image."""
        image_np = np.array(image)
        hist_r = np.histogram(image_np[:, :, 0], bins=16, range=(0, 256))[0]
        hist_g = np.histogram(image_np[:, :, 1], bins=16, range=(0, 256))[0]
        hist_b = np.histogram(image_np[:, :, 2], bins=16, range=(0, 256))[0]

        # Normalize histograms
        hist_r = hist_r / np.sum(hist_r)
        hist_g = hist_g / np.sum(hist_g)
        hist_b = hist_b / np.sum(hist_b)

        # Concatenate histograms into a single vector
        color_histograms = np.concatenate((hist_r, hist_g, hist_b))

        return color_histograms

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
                _ = self.visualizer.visualize_results(
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


def showcase_augmentations():
    image = Image.open("data/test_bg_removed/IMG_6944.png")
    augmentations = [
        image,
        image.rotate(90, expand=True, fillcolor="white"),
        image.rotate(180, expand=True, fillcolor="white"),
        image.rotate(270, expand=True, fillcolor="white"),
        image.transpose(Image.FLIP_LEFT_RIGHT),  # type: ignore
        image.transpose(Image.FLIP_TOP_BOTTOM),  # type: ignore
    ]

    plt.figure(figsize=(15, 10))
    for i, aug_image in enumerate(augmentations):
        plt.subplot(2, 3, i + 1)
        plt.imshow(aug_image)
        plt.title(f"Augmentation {i + 1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def showcase_histogram_colors_augmentation():
    image = Image.open("data/test_bg_removed/IMG_6944.png")
    """Calculate color histograms for the image."""
    image_np = np.array(image)
    hist_r = np.histogram(image_np[:, :, 0], bins=16, range=(0, 256))[0]
    hist_g = np.histogram(image_np[:, :, 1], bins=16, range=(0, 256))[0]
    hist_b = np.histogram(image_np[:, :, 2], bins=16, range=(0, 256))[0]

    # Normalize histograms
    hist_r = hist_r / np.sum(hist_r)
    hist_g = hist_g / np.sum(hist_g)
    hist_b = hist_b / np.sum(hist_b)

    # Concatenate histograms into a single vector
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.bar(range(16), hist_r, color="red")
    plt.title("Red Histogram")
    plt.subplot(1, 3, 2)
    plt.bar(range(16), hist_g, color="green")
    plt.title("Green Histogram")
    plt.subplot(1, 3, 3)
    plt.bar(range(16), hist_b, color="blue")
    plt.title("Blue Histogram")
    plt.tight_layout()
    plt.show()


def showcase_category_grid(csv_path, dam_dir, categories=None, samples_per_category=3):
    """
    Display a grid of DAM images in a single plot, organized by category.

    Args:
        csv_path (str): Path to CSV with image metadata
        dam_dir (str): Path to directory containing DAM images
        categories (list): List of categories to show (None for all)
        samples_per_category (int): Number of images per category to display
    """
    # Read category mappings from CSV
    mmc_to_category = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mmc = row["MMC"]
            category = row["Product_BusinessUnitDesc"]
            mmc_to_category[mmc] = category

    # Map DAM images to categories
    category_images = defaultdict(list)
    for filename in os.listdir(dam_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            mmc = os.path.splitext(filename)[0]
            if mmc in mmc_to_category:
                category = mmc_to_category[mmc]
                category_images[category].append(filename)

    # Determine which categories to display
    display_categories = categories or category_images.keys()
    valid_categories = [c for c in display_categories if category_images.get(c)]

    if not valid_categories:
        print("No images found for specified categories")
        return

    # Create figure grid
    n_rows = len(valid_categories)
    n_cols = samples_per_category
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    # Plot images in grid
    for row_idx, category in enumerate(valid_categories):
        images = category_images[category][:samples_per_category]

        for col_idx in range(n_cols):
            ax = axs[row_idx, col_idx] if n_rows > 1 else axs[col_idx]
            ax.axis("off")

            if col_idx < len(images):
                try:
                    img_path = os.path.join(dam_dir, images[col_idx])
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.set_title(images[col_idx], fontsize=8)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error\n{str(e)}", ha="center")

            # Add category label to first column
            if col_idx == 0:
                ax.text(
                    -0.3,
                    0.5,
                    category,
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=12,
                    rotation=0,
                )

    plt.tight_layout()
    plt.show()


def evaluate_vit_classification(searcher):
    df = pd.read_csv("data/labels.csv")
    df = df.rename(columns={"label_image_name": "MMC"})
    df["test_image_name"] = df["test_image_name"].apply(lambda x: x.split(".")[0])

    categories = pd.read_csv("data/product_list.csv")
    categories["MMC"] = categories["MMC"] + ".jpeg"

    df = df.merge(categories, on="MMC", how="left")

    correct_predictions = 0
    total_predictions = 0

    for image_path in tqdm(os.listdir("data/test_bg_removed")):
        similar_images = searcher.find_similar_images(
            "data/test_bg_removed/" + image_path
        )
        if not similar_images:
            continue

        similar_image_path = similar_images[0][0].split("/")[-1]

        predicted_category_row = df[df["MMC"] == similar_image_path]
        true_category_row = df[df["test_image_name"] == image_path.split(".")[0]]

        if not predicted_category_row.empty and not true_category_row.empty:
            predicted_category = predicted_category_row[
                "Product_BusinessUnitDesc"
            ].values[0]
            true_category = true_category_row["Product_BusinessUnitDesc"].values[0]
        else:
            continue

        if predicted_category == true_category:
            correct_predictions += 1
        total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Classification Accuracy: {accuracy:.4f}")


def calculate_hits(result_csv_name):
    # Read test images labels to build mappings
    test_image_to_category = {}
    mmc_to_category = {}
    with open("data/test_images_labels_with_categories.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Process test_image_name by removing extension
            test_image_name = row["test_image_name"].split(".")[0]
            mmc = row["MMC"]
            category = row["Product_BusinessUnitDesc"]
            test_image_to_category[test_image_name] = category
            # Build MMC to category mapping
            mmc_to_category[mmc] = category  # assumes MMC is unique per category

    # Read results and compute hits per category
    category_stats = defaultdict(
        lambda: {"total": 0, "hits1": 0, "hits3": 0, "hits5": 0, "hits10": 0}
    )
    with open(result_csv_name, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_image = row["test_image"]
            test_image_wo_ext = test_image.split(".")[0]
            # Skip if test image not found in labels
            if test_image_wo_ext not in test_image_to_category:
                continue
            true_category = test_image_to_category[test_image_wo_ext]

            # Extract similar images (up to 20 as per headers)
            similar_images = [row[f"similar_image{i}"] for i in range(20)]

            # Initialize hit flags for each k
            hit_1, hit_3, hit_5, hit_10 = False, False, False, False

            # Check hits for each k value
            for k in [1, 3, 5, 10]:
                top_k_images = similar_images[:k]
                categories_in_top_k = []
                for img in top_k_images:
                    mmc = img.split(".")[0]
                    category = mmc_to_category.get(mmc, None)
                    categories_in_top_k.append(category)
                # Check if true_category is in the list
                if true_category in categories_in_top_k:
                    if k == 1:
                        hit_1 = True
                    elif k == 3:
                        hit_3 = True
                    elif k == 5:
                        hit_5 = True
                    elif k == 10:
                        hit_10 = True

            # Update the category statistics
            category_stats[true_category]["total"] += 1
            if hit_1:
                category_stats[true_category]["hits1"] += 1
            if hit_3:
                category_stats[true_category]["hits3"] += 1
            if hit_5:
                category_stats[true_category]["hits5"] += 1
            if hit_10:
                category_stats[true_category]["hits10"] += 1

    # Calculate the hit percentages for each category
    result = {}
    for category, stats in category_stats.items():
        total = stats["total"]
        if total == 0:
            continue  # avoid division by zero
        hits1 = (stats["hits1"] / total) * 100
        hits3 = (stats["hits3"] / total) * 100
        hits5 = (stats["hits5"] / total) * 100
        hits10 = (stats["hits10"] / total) * 100
        result[category] = {
            "hits@1": hits1,
            "hits@3": hits3,
            "hits@5": hits5,
            "hits@10": hits10,
        }

    return result


def main():
    try:
        # Initialize the search system with experiment name
        experiment_name = "bg_removed_evaluation"
        searcher = ImageSimilaritySearch(experiment_name=experiment_name)
        visualizer = SimilarityVisualizer()

        # --------- Preprocessing (once only) ---------
        # Remove duplicates
        # from remove_duplicates import remove_duplicates
        # remove_duplicates("data/DAM", "data/labels.csv")

        # Preprocess images with background removal

        # Process catalog images
        # from background_removal import remove_backgrounds
        # remove_backgrounds(
        #     input_folder="data/DAM",
        #     output_folder="data/DAM_bg_removed",
        # )

        # # Process test images
        # remove_backgrounds(
        #     input_folder="data/test_image_headmind",
        #     output_folder="data/test_bg_removed",
        # )
        # --------- End of preprocessing ---------

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
