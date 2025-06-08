import torch

from lung_cancer_detection.model import ConvolutionalNetwork


def test_model_prediction():
    model = ConvolutionalNetwork(2)
    input = torch.rand(1, 3, 224, 224)

    prediction = model(input)
    assert prediction.shape == (1, 3), "Output should be (batch_size, num_classes)!"
    assert torch.allclose(
        prediction.sum(), torch.tensor(1.0)
    ), "Predictions should sum to 1 (softmax)!"
