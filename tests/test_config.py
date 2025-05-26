def config_creation():
    return {
        "data_loading": {"data_path": "lung_image_sets", "image_size": 254},
        "training": {"batch_size": 2, "num_workers": 7},
    }


def test_config(data_regression):
    config = config_creation()
    data_regression.check(config)
