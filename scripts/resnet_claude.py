import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm


class ProductImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [
            f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name


class DiorImageMatcher:
    def __init__(
        self,
        dam_dir,
        test_dir,
        product_list_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.dam_dir = dam_dir
        self.test_dir = test_dir
        self.device = device

        # Load product categories
        self.product_df = pd.read_csv(product_list_path)

        # Initialize model and transforms
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()  # type: ignore # Remove classification layer
        self.model = self.model.to(device)
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

        # Initialize datasets
        self.dam_dataset = ProductImageDataset(dam_dir, self.transform)
        self.test_dataset = ProductImageDataset(test_dir, self.transform)

        # Create dataloaders
        self.dam_loader = DataLoader(self.dam_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        # Store features
        self.dam_features = None
        self.dam_filenames = None

    def extract_features(self, dataloader):
        features = []
        filenames = []

        with torch.no_grad():
            for images, batch_filenames in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)
                batch_features = self.model(images)
                features.append(batch_features.cpu().numpy())
                filenames.extend(batch_filenames)

        return np.vstack(features), filenames

    def build_catalog_index(self):
        """Extract and store features for all catalog images"""
        print("Building catalog index...")
        self.dam_features, self.dam_filenames = self.extract_features(self.dam_loader)
        print(f"Indexed {len(self.dam_filenames)} catalog images")

    def find_matches(self, test_features, top_k=5):
        """Find top-k matches for each test image"""
        similarities = cosine_similarity(test_features, self.dam_features)
        top_matches_idx = np.argsort(-similarities, axis=1)[:, :top_k]

        matches = []
        for i in range(len(top_matches_idx)):
            match_info = []
            for idx in top_matches_idx[i]:
                match_filename = self.dam_filenames[idx]  # type: ignore
                similarity_score = similarities[i][idx]
                category = self.get_product_category(match_filename)
                match_info.append(
                    {
                        "catalog_image": match_filename,
                        "similarity_score": similarity_score,
                        "category": category,
                    }
                )
            matches.append(match_info)

        return matches

    def get_product_category(self, filename):
        """Get product category from filename using product_list.csv"""
        # Implement category lookup based on your CSV structure
        # This is a placeholder - adjust according to your actual data structure
        match = self.product_df[self.product_df["MMC"] == filename]
        if not match.empty:
            return match.iloc[0]["Product_BusinessUnitDesc"]
        return "Unknown"

    def match_test_images(self, output_path):
        """Match all test images to catalog images and save results"""
        if self.dam_features is None:
            self.build_catalog_index()

        print("Processing test images...")
        test_features, test_filenames = self.extract_features(self.test_loader)
        matches = self.find_matches(test_features)

        # Prepare results for CSV
        results = []
        for test_img, match_list in zip(test_filenames, matches):
            top_match = match_list[0]  # Get the best match
            results.append(
                {
                    "test_image_name": test_img,
                    "label_image_name": top_match["catalog_image"],
                    "confidence_score": top_match["similarity_score"],
                    "category": top_match["category"],
                }
            )

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        return results_df


def evaluate_accuracy(predictions_path, ground_truth_path):
    """Evaluate the accuracy of predictions against ground truth labels"""
    predictions = pd.read_csv(predictions_path)
    ground_truth = pd.read_csv(ground_truth_path)

    # Merge predictions with ground truth
    merged = predictions.merge(
        ground_truth, on="test_image_name", suffixes=("_pred", "_true")
    )

    # Calculate accuracy
    correct = (merged["label_image_name_pred"] == merged["label_image_name_true"]).sum()
    total = len(merged)
    accuracy = correct / total

    print(f"Accuracy: {accuracy:.2%}")
    return accuracy


# Initialize the matcher
matcher = DiorImageMatcher(
    dam_dir="data/DAM",
    test_dir="data/test_image_headmind",
    product_list_path="data/product_list.csv",
)

# Run matching and save results
results = matcher.match_test_images("predictions.csv")

# Evaluate accuracy
accuracy = evaluate_accuracy("predictions.csv", "data/labels.csv")
