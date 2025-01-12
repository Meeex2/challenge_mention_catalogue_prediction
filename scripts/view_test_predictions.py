import math
import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class ImagePredictor:
    def __init__(
        self,
        model_path: str,
        num_classes: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model = self._load_model(model_path, num_classes)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_model(self, model_path: str, num_classes: int) -> nn.Module:
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def predict_image(
        self, image: Image.Image, categories: List[str]
    ) -> Dict[str, float]:
        """Predict category probabilities for a single image."""
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # type: ignore

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        predictions = {
            categories[i]: prob.item() for i, prob in enumerate(probabilities)
        }
        return dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))


def display_image_grid(images, predictions, titles, grid_size=4):
    """Display images in a grid with their predictions."""
    n_images = len(images)
    rows = math.ceil(n_images / grid_size)
    cols = min(n_images, grid_size)

    plt.figure(figsize=(15, 3 * rows))

    for idx, (img, pred, title) in enumerate(zip(images, predictions, titles)):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        # Show top 3 predictions
        top_3 = list(pred.items())[:3]
        pred_text = "\n".join([f"{cat}: {prob:.1%}" for cat, prob in top_3])
        plt.title(f"{title}\n\n{pred_text}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # Configuration
    MODEL_PATH = "./models/best_model.pth"
    TEST_DIR = "./data/test_image_headmind"
    CSV_PATH = "./data/product_list.csv"
    GRID_SIZE = 4  # Number of images per row
    NUM_IMAGES = 8  # Number of images to display
    RANDOM_SEED = 4  # For reproducibility

    random.seed(RANDOM_SEED)

    # Load category mapping
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    categories = sorted(df["Product_BusinessUnitDesc"].unique())

    # Initialize predictor
    predictor = ImagePredictor(MODEL_PATH, len(categories))
    print(f"Loaded model from {MODEL_PATH}")

    # Get list of test images and randomly sample
    all_test_images = [
        f for f in os.listdir(TEST_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    print(f"Found {len(all_test_images)} total test images")

    selected_images = random.sample(
        all_test_images, min(NUM_IMAGES, len(all_test_images))
    )
    print(f"Displaying {len(selected_images)} random images")

    images = []
    predictions = []
    titles = []

    # Load and predict for selected images
    for image_file in selected_images:
        image_path = os.path.join(TEST_DIR, image_file)
        image = Image.open(image_path).convert("RGB")
        pred = predictor.predict_image(image, categories)

        images.append(image)
        predictions.append(pred)
        titles.append(image_file)

    # Display images in grid
    display_image_grid(images, predictions, titles, GRID_SIZE)


if __name__ == "__main__":
    main()
