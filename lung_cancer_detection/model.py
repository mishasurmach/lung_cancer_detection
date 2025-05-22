from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(pl.LightningModule):
    """Convolutional neural network for multi-class image classification.

    This model consists of two convolutional layers followed by four fully
    connected layers.It is designed for classification tasks with three
    output classes and RGB images.
    """

    def __init__(self, lr):
        """Initializes the convolutional network.

        Args:
            config.
        """
        super().__init__()
        self.save_hyperparameters(lr)
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, 3)
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, X):
        """Performs a forward pass through the network.

        Args:
            X (torch.Tensor): Input tensor of shape (B, 3, H, W), where B is batch size.

        Returns:
            torch.Tensor: Log-probabilities for each class, shape (B, 3).
        """
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0) -> float:
        """Computes and logs the training loss and accuracy.

        Args:
            batch (Any): A batch of data in the form (inputs, targets).
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader (default is 0).

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        X, y = batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Computes and logs validation loss and accuracy.

        Args:
            batch (Any): A batch of data in the form (inputs, targets).
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader (default is 0).
        """
        X, y = batch
        y_hat = self(X)
        loss = self.loss_fn(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Computes and logs test loss and accuracy.

        Args:
            batch (Any): A batch of data in the form (inputs, targets).
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader (default is 0).
        """
        X, y = batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Performs a forward pass during prediction.

        Args:
            batch (Any): A batch of input data.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): Index of the current dataloader
            (for multiple dataloaders).

        Returns:
            Tensor: The predicted class indices for the batch.
        """
        X, _ = batch
        y_hat = self(X)
        preds = y_hat.argmax(dim=1)
        return preds

    def configure_optimizers(self) -> Any:
        """Configures the optimizer.

        Returns:
            torch.optim.Optimizer: Adam optimizer initialized with model parameters.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
