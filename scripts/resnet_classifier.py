import logging
import os
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DAMDataset(Dataset):
    """Custom dataset for Digital Asset Management (DAM) image classification."""

    def __init__(
        self,
        image_dir: str,
        mmc_to_category: Dict[str, str],
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mmc_to_category = mmc_to_category
        self.transform = transform
        self.categories = list(set(self.mmc_to_category.values()))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_file = self.image_files[idx]
        mmc_code = os.path.splitext(image_file)[0]
        category = self.mmc_to_category.get(mmc_code, "Unknown")
        category_idx = self.categories.index(category)

        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, category_idx  # type: ignore


class CategoryClassifier:
    """ResNet-based image classifier for product categories with version selection."""

    def __init__(
        self,
        num_classes: int,
        resnet_version: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_save_path: str = "resnet_category_classifier.pth",
    ):
        self.device = device
        self.resnet_version = resnet_version
        self.model_save_path = model_save_path
        self.model = self._initialize_model(num_classes)
        self._load_model_if_exists()
        logging.info(
            f"Initialized ResNet-{resnet_version} classifier with {num_classes} classes on {device}"
        )

    def _initialize_model(self, num_classes: int) -> nn.Module:
        # Load pretrained ResNet with specified version
        if self.resnet_version == 50:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        elif self.resnet_version == 101:
            weights = models.ResNet101_Weights.DEFAULT
            model = models.resnet101(weights=weights)
        elif self.resnet_version == 152:
            weights = models.ResNet152_Weights.DEFAULT
            model = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet version: {self.resnet_version}")

        # Freeze all parameters except final layer
        for param in model.parameters():
            param.requires_grad = False

        # Replace final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        return model.to(self.device)

    def _load_model_if_exists(self):
        """Load the model if a saved model exists."""
        if os.path.exists(self.model_save_path):
            self.model.load_state_dict(
                torch.load(self.model_save_path, weights_only=True)
            )
            logging.info(f"Loaded model from {self.model_save_path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2
        )

        best_val_loss = float("inf")

        for epoch in tqdm(range(num_epochs), desc="Training"):
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            val_loss, accuracy = self._validate_epoch(val_loader, criterion)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_path)
                logging.info(
                    f"Saved new best model with validation loss: {val_loss:.4f}"
                )

            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Accuracy: {accuracy * 100:.4f}%"
            )

    def _train_epoch(
        self, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer
    ) -> float:
        self.model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(
        self, val_loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                total_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return total_loss / len(val_loader), accuracy

    def predict(self, image: torch.Tensor) -> int:
        """
        Make a prediction for a single image.

        Args:
            image: Input image tensor

        Returns:
            Predicted category index
        """
        self.model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)
            output = self.model(image)
            _, predicted = torch.max(output.data, 1)
            return int(predicted.item())


def create_train_val_datasets(
    dataset: Dataset, val_size: float = 0.2, random_state: int = 42
) -> Tuple[Subset, Subset]:
    """
    Split a dataset into training and validation sets.

    Args:
        dataset: The full dataset to split
        val_size: Proportion of the dataset to use for validation
        random_state: Random seed for reproducibility

    Returns:
        A tuple of (train_dataset, val_dataset)
    """
    indices = list(range(len(dataset)))  # type: ignore
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_size,
        random_state=random_state,
        stratify=[dataset[i][1] for i in indices],  # Stratify by labels
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    logging.info(
        f"Split dataset into {len(train_dataset)} training and {len(val_dataset)} validation samples"
    )
    return train_dataset, val_dataset


def get_transforms(is_training: bool = True) -> transforms.Compose:
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if is_training:
        base_transforms.insert(1, transforms.RandomHorizontalFlip())
        base_transforms.insert(2, transforms.ColorJitter(brightness=0.2, contrast=0.2))

    return transforms.Compose(base_transforms)


def evaluate_model(classifier: CategoryClassifier, val_loader: DataLoader) -> None:
    y_true = []
    y_pred = []

    classifier.model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = (
                images.to(classifier.device),
                labels.to(classifier.device),
            )
            outputs = classifier.model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    f1 = f1_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)

    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Accuracy: {accuracy:.4f}")


def main():
    # Configuration
    RESNET_VERSION = 50  # Change to 50, 101, or 152 as needed
    DATA_DIR = "./data"
    DAM_DIR = os.path.join(DATA_DIR, "DAM")
    CSV_PATH = os.path.join(DATA_DIR, "product_list.csv")
    MODEL_SAVE_PATH = os.path.join("models", f"resnet_{RESNET_VERSION}_model.pth")
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    VAL_SIZE = 0.2
    RANDOM_STATE = 42

    # Data loading and preprocessing
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    mmc_to_category = dict(zip(df["MMC"], df["Product_BusinessUnitDesc"]))
    categories = sorted(df["Product_BusinessUnitDesc"].unique())

    full_dataset = DAMDataset(
        image_dir=DAM_DIR,
        mmc_to_category=mmc_to_category,
        transform=get_transforms(is_training=True),
    )

    train_dataset, val_dataset = create_train_val_datasets(
        full_dataset, val_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize and train classifier
    classifier = CategoryClassifier(
        num_classes=len(categories),
        resnet_version=RESNET_VERSION,
        model_save_path=MODEL_SAVE_PATH,
    )
    classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
    )

    logging.info("Training completed successfully")

    # Load the best model and evaluate
    classifier.model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    evaluate_model(classifier, train_loader)


if __name__ == "__main__":
    main()
