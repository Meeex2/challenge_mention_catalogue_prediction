import logging
import os
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
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
        """
        Args:
            image_dir: Directory containing the images
            mmc_to_category: Mapping from MMC codes to category names
            transform: Optional transforms to be applied to images
        """
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
            image = self.transform(image)  # transform will handle conversion to tensor
        else:
            # If no transform is provided, at least convert to tensor
            image = transforms.ToTensor()(image)

        return image, category_idx  # type: ignore


class CategoryClassifier:
    """VGG16-based image classifier for product categories."""

    def __init__(
        self,
        num_classes: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            num_classes: Number of product categories
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = self._initialize_model(num_classes)
        logging.info(
            f"Initialized CategoryClassifier with {num_classes} classes on {device}"
        )

    def _initialize_model(self, num_classes: int) -> nn.Module:
        model = models.vgg16(weights="VGG16_Weights.DEFAULT")
        # Freeze early layers
        for param in list(model.parameters())[:-4]:
            param.requires_grad = False
        # Replace classification head
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        model_save_path: str = "best_model.pth",
    ) -> None:
        """
        Train the classifier.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            model_save_path: Path to save the best model
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2
        )

        best_val_loss = float("inf")

        for epoch in tqdm(range(num_epochs), desc="Training"):
            # Training phase
            train_loss = self._train_epoch(train_loader, criterion, optimizer)

            # Validation phase
            val_loss = self._validate_epoch(val_loader, criterion)

            # Update learning rate
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_save_path)
                logging.info(
                    f"Saved new best model with validation loss: {val_loss:.4f}"
                )

            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f}"
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

    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                total_loss += criterion(outputs, labels).item()

        return total_loss / len(val_loader)

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


def get_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Get image transforms for training or validation.

    Args:
        is_training: Whether to include data augmentation transforms

    Returns:
        Composition of transforms
    """
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if is_training:
        base_transforms.insert(1, transforms.RandomHorizontalFlip())
        base_transforms.insert(2, transforms.ColorJitter(brightness=0.2, contrast=0.2))

    return transforms.Compose(base_transforms)


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


def main():
    # Configuration
    DATA_DIR = "./data"
    DAM_DIR = os.path.join(DATA_DIR, "DAM")
    CSV_PATH = os.path.join(DATA_DIR, "product_list.csv")
    MODEL_SAVE_PATH = os.path.join("models", "best_model.pth")
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    VAL_SIZE = 0.2
    RANDOM_STATE = 42

    # Load and prepare data
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    mmc_to_category = dict(zip(df["MMC"], df["Product_BusinessUnitDesc"]))
    categories = sorted(df["Product_BusinessUnitDesc"].unique())

    # Create full dataset
    full_dataset = DAMDataset(
        image_dir=DAM_DIR,
        mmc_to_category=mmc_to_category,
        transform=get_transforms(
            is_training=True
        ),  # We'll apply specific transforms later
    )

    # Split dataset
    train_dataset, val_dataset = create_train_val_datasets(
        full_dataset, val_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # Apply appropriate transforms
    train_dataset.dataset.transform = get_transforms(is_training=True)  # type: ignore
    val_dataset.dataset.transform = get_transforms(is_training=False)  # type: ignore

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize and train classifier
    classifier = CategoryClassifier(num_classes=len(categories))
    classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        model_save_path=MODEL_SAVE_PATH,
    )

    logging.info("Training completed successfully")


if __name__ == "__main__":
    main()
