"""
Code given by ChatGPT
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# Directories
DAM_DIR = "data/DAM"
TEST_DIR = "data/test_bg_removed"
PRODUCT_LIST_CSV = "data/product_list.csv"
LABELS_CSV = "data/labels.csv"

# Load mappings
product_list = pd.read_csv(PRODUCT_LIST_CSV)
labels = pd.read_csv(LABELS_CSV)

# Preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, self.image_names[idx]


# Feature extraction using ResNet
class FeatureExtractor:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(
            *list(self.model.children())[:-1]
        )  # Remove classification head
        self.model.eval()

    def extract_features(self, dataloader):
        features = []
        image_names = []
        with torch.no_grad():
            for images, names in tqdm(dataloader):
                outputs = (
                    self.model(images).squeeze(-1).squeeze(-1)
                )  # Global avg pooling
                features.append(outputs.numpy())
                image_names.extend(names)
        return np.vstack(features), image_names


# Create data loaders
catalog_dataset = ImageDataset(DAM_DIR, transform)
test_dataset = ImageDataset(TEST_DIR, transform)

catalog_loader = DataLoader(catalog_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Extract features
extractor = FeatureExtractor()
catalog_features, catalog_names = extractor.extract_features(catalog_loader)
test_features, test_names = extractor.extract_features(test_loader)

# Map catalog images to categories
catalog_categories = product_list.set_index("MMC").to_dict()["Product_BusinessUnitDesc"]
catalog_category_map = {
    name: catalog_categories.get(name.split(".")[0], None) for name in catalog_names
}

# Match test images to catalog images
results = []
for i, test_feature in enumerate(test_features):
    test_name = test_names[i]
    # Find category of the test image from labels.csv
    test_category_row = labels.loc[
        labels["test_image_name"] == test_name, "label_image_name"
    ]
    if test_category_row.empty:
        print(f"Problem: No category found for test image {test_name}")
        continue
    test_category = test_category_row.values[0]  # type: ignore
    test_category = catalog_category_map[test_category]

    # Filter catalog features by category
    filtered_features = [
        (feature, name)
        for feature, name in zip(catalog_features, catalog_names)
        if catalog_category_map[name] == test_category
    ]
    filtered_features, filtered_names = zip(*filtered_features)
    filtered_features = np.vstack(filtered_features)

    # Compute similarity
    similarities = cosine_similarity(test_feature.reshape(1, -1), filtered_features)
    sorted_indices = np.argsort(-similarities[0])  # Sort in descending order
    sorted_names = [filtered_names[idx] for idx in sorted_indices]

    results.append((test_name, sorted_names))

# Save results
results_df = pd.DataFrame(
    [(test_name, names[0]) for test_name, names in results],
    columns=["test_image_name", "predicted_label_image_name"],
)
results_df.to_csv("data/predictions.csv", index=False)

# Evaluate results
hits_at_1 = 0
hits_at_3 = 0
hits_at_5 = 0
hits_at_10 = 0
reciprocal_ranks = []

for test_name, predicted_names in results:
    true_label = labels.loc[
        labels["test_image_name"] == test_name, "label_image_name"
    ].values[0]  # type: ignore
    if true_label in predicted_names:
        rank = predicted_names.index(true_label) + 1
        reciprocal_ranks.append(1 / rank)
        if rank == 1:
            hits_at_1 += 1
        if rank <= 3:
            hits_at_3 += 1
        if rank <= 5:
            hits_at_5 += 1
        if rank <= 10:
            hits_at_10 += 1

# Normalize metrics
hits_at_1 = hits_at_1 / len(results)
hits_at_3 = hits_at_3 / len(results)
hits_at_5 = hits_at_5 / len(results)
hits_at_10 = hits_at_10 / len(results)
mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0

print(f"Hits@1: {hits_at_1:.2%}")
print(f"Hits@3: {hits_at_3:.2%}")
print(f"Hits@5: {hits_at_5:.2%}")
print(f"Hits@10: {hits_at_10:.2%}")
print(f"Mean Reciprocal Rank (mRR): {mrr:.4f}")
