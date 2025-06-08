import hydra
import pytorch_lightning as pl
from dataset import DataModule
from model import ConvolutionalNetwork
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    pl.seed_everything(42)
    dm = DataModule(config)
    model = ConvolutionalNetwork(
        lr=config["training"]["lr"],
    )

    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=config["logging"]["experiment_name"],
            run_name=config["logging"]["run_name"],
            save_dir=config["logging"]["mlflow_save_dir"],
            tracking_uri=config["logging"]["tracking_uri"],
        )
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=2),
    ]

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=config["model"]["model_local_path"],
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=config["model"]["save_top_k"],
            every_n_epochs=config["model"]["every_n_epochs"],
        )
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
