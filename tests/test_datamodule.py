import pytest

from lung_cancer_detection.dataset import DataModule


@pytest.fixture
def datamodule():
    test_config = {
        "data_loading": {"data_path": "lung_image_sets", "image_size": 254},
        "training": {"batch_size": 2, "num_workers": 7},
    }

    dm = DataModule(test_config)
    dm.setup()
    return dm


@pytest.mark.requires_files
def test_dataloaders(datamodule):
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    images, labels = batch
    assert images.shape[0] == 2, "First shape should be batch_size!"
    assert len(labels) == 2, "There should be only batch_size labels"

    assert set(labels.numpy()) <= {0, 1, 2}, "Class should be one of {0, 1, 2}!"
