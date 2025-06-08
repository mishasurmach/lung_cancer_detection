import hydra
import pytorch_lightning as pl
from dataset import DataModule
from model import ConvolutionalNetwork
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    dm = DataModule(config)
    dm.setup()
    test_dataloader = dm.test_dataloader()
    model = ConvolutionalNetwork.load_from_checkpoint(config["inference"]["ckpt_path"])
    trainer = pl.Trainer(accelerator="gpu", devices="auto")
    trainer.test(model=model, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
