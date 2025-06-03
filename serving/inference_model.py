import os

import mlflow.pyfunc
import torch
import torchvision.transforms as transforms
from PIL import Image

from lung_cancer_detection.model import ConvolutionalNetwork


class CancerClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
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
        images = []
        for img_data in model_input:
            img_path = str(img_data["data"])

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
