import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.distance import cosine
from torchvision import models, transforms
from tqdm import tqdm

# Load CSV files
product_list = pd.read_csv("data/product_list.csv")
labels = pd.read_csv("data/labels.csv")

# Define image transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_images_from_folder(folder):
    images = []
    for filename in tqdm(os.listdir(folder), desc=f"Loading images from {folder}"):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        images.append(img)
    return images


# Load catalog and test images
catalog_images = load_images_from_folder("data/DAM")
test_images = load_images_from_folder("data/test_image_headmind")

# Load the ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Remove the final classification layer
model = torch.nn.Sequential(*(list(model.children())[:-1]))


def extract_features(images):
    features = []
    for img in tqdm(images, desc="Extracting features"):
        img = img.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            feat = model(img)
        feat = feat.squeeze().numpy()
        features.append(feat)
    return features


# Extract features for catalog and test images
catalog_features = extract_features(catalog_images)
test_features = extract_features(test_images)


def find_most_similar(test_feature, catalog_features):
    similarities = []
    for catalog_feature in catalog_features:
        similarity = 1 - cosine(test_feature, catalog_feature)
        similarities.append(similarity)
    most_similar_index = np.argmax(similarities)
    return most_similar_index


results = []
for i, test_feature in enumerate(
    tqdm(test_features, desc="Finding most similar images")
):
    most_similar_index = find_most_similar(test_feature, catalog_features)
    results.append((i, most_similar_index))

# Evaluate the results
correct_matches = 0
for i, most_similar_index in results:
    predicted_label = product_list.iloc[most_similar_index]["MMC"]
    true_label = labels.iloc[i]["label_image_name"]
    if predicted_label == true_label:
        correct_matches += 1

accuracy = correct_matches / len(results)
print(f"Accuracy: {accuracy * 100:.2f}%")
