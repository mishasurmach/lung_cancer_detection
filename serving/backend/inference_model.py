import os

import mlflow.pyfunc
import torch
import torchvision.transforms as transforms
from PIL import Image

from lung_cancer_detection.model import ConvolutionalNetwork


class CancerClassifier(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper for lung cancer image classification model.

    This class defines how the PyTorch Lightning model is loaded and
    how inference is performed on input image paths received via MLflow's
    REST API.
    """

    def load_context(self, context):
        """Loads model weights and preprocessing pipeline at serving time.

        This method is called once when the MLflow model is loaded.
        It restores the trained model from checkpoint and prepares
        the image transformation pipeline.

        Args:
            context (mlflow.pyfunc.PythonModelContext): Contains model
            artifacts and other metadata.
        """
        self.device = torch.device("cpu")

        self.model = ConvolutionalNetwork.load_from_checkpoint(
            checkpoint_path=context.artifacts["model_weights"], lr=1e-3
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, context, model_input):
        """Performs inference on a batch of image file paths.

        The input is expected to be a pandas DataFrame with a column `"data"` containing
        absolute file paths to the images. The method loads and transforms each image,
        stacks them into a batch, and returns the predicted class indices.

        Args:
            context (mlflow.pyfunc.PythonModelContext): MLflow context object.
            model_input (pd.DataFrame): A DataFrame with a column `"data"`
            of image paths.

        Returns:
            np.ndarray: Array of predicted class indices for each input image.
        """
        images = []
        for _, row in model_input.iterrows():
            img_path = row["data"]

            if not os.path.exists(img_path):
                raise ValueError(f"File {img_path} not found")

            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0)
            images.append(img_tensor)

        batch = torch.cat(images, dim=0).to(self.device)

        with torch.no_grad():
            preds = self.model(batch).argmax(dim=1)

        return preds.cpu().numpy()


if __name__ == "__main__":
    mlflow.pyfunc.save_model(
        path="lung_cancer_model",
        python_model=CancerClassifier(),
        artifacts={"model_weights": "best_model.ckpt"},
    )
