from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class DataModule(pl.LightningDataModule):
    """LightningDataModule for image classification using ImageFolder.

    Loads images from a single root directory, applies standard
    transformations,and splits the dataset into stratified train,
    validation, and test subsets.
    """

    def __init__(self, config):
        """Initializes the data module.

        Args:
            config (dict): Configuration dictionary.
        """
        super().__init__()
        self.config = config
        self.transform = transforms.Compose(
            [
                # rotate +/- 10 degrees
                transforms.RandomRotation(10),
                # reverse 50% of images
                transforms.RandomHorizontalFlip(),
                # resize shortest side to 224 pixels
                transforms.Resize(224),
                # crop longest side to 224 pixels at center
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Prepares the dataset.

        This method is intended for one-time operations such as downloading or
        unzipping files. It is called only on a single process in distributed
        training.

        In this implementation, it does nothing since the dataset is assumed to
        already exist locally.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Sets up training, validation, and test datasets.

        Loads the dataset using ImageFolder and splits it into training,
        validation, and test sets while preserving the class distribution
        (stratified split). This method is called by Lightning at different
        stages of the training lifecycle.

        Args:
            stage (Optional[str]): One of {"fit", "validate", "test",
                "predict"}.
                Determines which datasets to prepare. If None, all datasets are
                prepared.
        """
        if self.dataset is None:
            self.dataset = datasets.ImageFolder(
                root=self.config["data_loading"]["data_path"], transform=self.transform
            )
            targets = np.array([sample[1] for sample in self.dataset.samples])
            indices = np.arange(len(targets))

            train_idx, temp_idx, _, temp_labels = train_test_split(
                indices, targets, test_size=0.2, stratify=targets, random_state=42
            )

            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
            )

            self.train_dataset = Subset(self.dataset, train_idx)
            self.val_dataset = Subset(self.dataset, val_idx)
            self.test_dataset = Subset(self.dataset, test_idx)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the training DataLoader.

        Returns:
            DataLoader: PyTorch DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the validation DataLoader.

        Returns:
            DataLoader: PyTorch DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset, batch_size=self.config["training"]["batch_size"]
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the test DataLoader.

        Returns:
            DataLoader: PyTorch DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset, batch_size=self.config["training"]["batch_size"]
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the prediction DataLoader.

        Note:
            In this implementation, the prediction dataset is the same as the
            test dataset.

        Returns:
            DataLoader: PyTorch DataLoader for the prediction dataset.
        """
        return DataLoader(
            self.test_dataset, batch_size=self.config["training"]["batch_size"]
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Cleans up after training, validation, testing, or prediction.

        Called at the end of each stage. No action is taken in this implementation.

        Args:
            stage (Optional[str]): One of {"fit", "validate", "test", "predict"}.
        """
        pass
