import numpy as np
import pytorch_lightning as pl
from dataset import DataModule
from model import ConvolutionalNetwork


def main(config):
    dm = DataModule(config).setup()
    test_dataloader = dm.test_dataloader()
    model = ConvolutionalNetwork.load_from_checkpoint(config["inference"]["ckpt_path"])
    trainer = pl.Trainer(accelerator="gpu", devices="auto")
    accs = trainer.test(model=model, dataloaders=test_dataloader)
    print(f"Test accuracy: {np.mean(accs):.2f}")


if __name__ == "__main__":
    main()
