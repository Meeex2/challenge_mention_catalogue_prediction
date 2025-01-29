import os

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Configuration
CFG = {
    "model_name": "tf_efficientnet_b4",
    "embedding_size": 512,
    "batch_size": 2,
    "num_workers": 4,
    "num_epochs": 15,
    "lr": 1e-4,
    "margin": 0.2,
    "image_size": 384,
    "category_weight": 0.3,
}

# Advanced augmentations
train_transform = transforms.Compose(
    [
        transforms.Resize((CFG["image_size"], CFG["image_size"])),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((CFG["image_size"], CFG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Custom dataset with triplet sampling
class TripletDataset(Dataset):
    def __init__(self, labels_df, catalog_features, catalog_categories, transform=None):
        self.pairs = list(
            zip(labels_df["test_image_name"], labels_df["label_image_name"])
        )
        self.catalog_features = catalog_features
        self.catalog_categories = catalog_categories
        self.transform = transform
        self.test_image_folder = "data/test_image_headmind"
        self.catalog_image_folder = "data/DAM"

        # Create category map
        self.category_map = {}
        for img, cat in zip(
            catalog_categories["MMC"], catalog_categories["category_idx"]
        ):
            self.category_map[img] = cat

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        test_img, pos_img = self.pairs[idx]
        test_cat = self.category_map[pos_img.split(".")[0]]

        # Load test image
        test_path = os.path.join(self.test_image_folder, test_img)
        test_image = Image.open(test_path).convert("RGB")

        # Load positive catalog image
        pos_path = os.path.join(self.catalog_image_folder, pos_img)
        pos_image = Image.open(pos_path).convert("RGB")

        # Find hard negative from same category
        same_cat_imgs = [
            img for img, cat in self.category_map.items() if cat == test_cat
        ]
        neg_img = np.random.choice([img for img in same_cat_imgs if img != pos_img])
        neg_path = os.path.join(self.catalog_image_folder, neg_img)
        neg_image = Image.open(neg_path + ".jpeg").convert("RGB")

        if self.transform:
            test_image = self.transform(test_image)
            pos_image = self.transform(pos_image)
            neg_image = self.transform(neg_image)

        return test_image, pos_image, neg_image


# Enhanced model with multi-head output
class CustomModel(nn.Module):
    def __init__(self, model_name, embedding_size, num_categories):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.embedding = nn.Linear(self.backbone.num_features, embedding_size)
        self.category_head = nn.Linear(self.backbone.num_features, num_categories)

    def forward(self, x):
        features = self.backbone(x).to(torch.float16)
        embedding = self.embedding(features)
        category = self.category_head(features)
        return embedding, category


# Triplet loss with category regularization
class EnhancedLoss(nn.Module):
    def __init__(self, margin=0.2, alpha=0.3):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        self.category_loss = nn.CrossEntropyLoss()

    def forward(self, anchor, positive, negative, cat_pred, cat_true):
        trip_loss = self.triplet_loss(anchor, positive, negative)
        cat_loss = self.category_loss(cat_pred, cat_true)
        return trip_loss + self.alpha * cat_loss


def train_model():
    # Load data
    product_df = pd.read_csv("data/product_list.csv")
    labels_df = pd.read_csv("data/labels.csv")

    # Prepare category labels
    category_mapping = {
        cat: idx
        for idx, cat in enumerate(product_df["Product_BusinessUnitDesc"].unique())
    }
    product_df["category_idx"] = product_df["Product_BusinessUnitDesc"].map(
        category_mapping
    )

    # Initialize model
    model = CustomModel(CFG["model_name"], CFG["embedding_size"], len(category_mapping))
    model = model.to(device).half()  # Use half precision

    # Dataset and loader
    dataset = TripletDataset(
        labels_df,
        product_df[["MMC"]],
        product_df[["MMC", "category_idx"]],
        train_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
    )

    # Loss and optimizer
    criterion = EnhancedLoss(margin=CFG["margin"], alpha=CFG["category_weight"])
    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"])

    # Training loop
    model.train()
    for epoch in range(CFG["num_epochs"]):
        total_loss = 0
        for anchor, pos, neg in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            anchor, pos, neg = (
                anchor.to(device).half(),
                pos.to(device).half(),
                neg.to(device).half(),
            )

            optimizer.zero_grad()

            anchor_emb, anchor_cat = model(anchor)
            pos_emb, _ = model(pos)
            neg_emb, _ = model(neg)

            # Get category labels
            _, cat_labels = torch.max(anchor_cat, 1)

            loss = criterion(anchor_emb, pos_emb, neg_emb, anchor_cat, cat_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")

    return model, category_mapping


# Feature extraction with category filtering
def enhanced_feature_extraction(
    model, image_folder, image_names, category_mapping, transform
):
    model.eval()
    device = next(model.parameters()).device

    features = {}
    categories = {}

    with torch.no_grad():
        for img_name in tqdm(image_names):
            img_path = os.path.join(image_folder, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0).to(device).half()
                emb, cat = model(img)
                features[img_name] = emb.squeeze().cpu().numpy()
                categories[img_name] = torch.argmax(cat).item()
            except:
                continue

    return features, categories


# Evaluation with category filtering
def evaluate_model(model, category_mapping):
    # Load data
    product_df = pd.read_csv("data/product_list.csv")
    labels_df = pd.read_csv("data/labels.csv")

    # Extract catalog features
    catalog_features, catalog_categories = enhanced_feature_extraction(
        model, "data/DAM", product_df["MMC"].tolist(), category_mapping, test_transform
    )

    # Extract test features
    test_features, test_categories = enhanced_feature_extraction(
        model,
        "data/test_image_headmind",
        labels_df["test_image_name"].tolist(),
        category_mapping,
        test_transform,
    )

    # Create category index
    category_index = {}
    for img, cat in catalog_categories.items():
        category_index.setdefault(cat, []).append(img)

    # Evaluate
    correct = 0
    for test_img, true_label in zip(
        labels_df["test_image_name"], labels_df["label_image_name"]
    ):
        if test_img not in test_features or true_label not in catalog_features:
            continue

        test_feat = test_features[test_img]
        test_cat = test_categories[test_img]

        # Filter catalog by category
        candidate_imgs = category_index.get(test_cat, [])
        if not candidate_imgs:
            candidate_imgs = list(catalog_features.keys())

        # Compute similarities
        candidate_feats = np.array([catalog_features[img] for img in candidate_imgs])
        distances = np.linalg.norm(candidate_feats - test_feat, axis=1)
        nearest_idx = np.argmin(distances)
        predicted_label = candidate_imgs[nearest_idx]

        if predicted_label == true_label:
            correct += 1

    accuracy = correct / len(labels_df)
    print(f"Enhanced Accuracy: {accuracy * 100:.2f}%")
    return accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    trained_model, category_mapping = train_model()

    # Evaluate
    evaluate_model(trained_model, category_mapping)
