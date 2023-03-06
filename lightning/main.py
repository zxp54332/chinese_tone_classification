# main.py
import warnings

import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import MLFlowLogger
from tone_pl_model import LightningToneClassifier

from tone_datamodule import ToneDataModule


def prepare_fit_dataloader(cli: LightningCLI):
    cli.datamodule.prepare_data()
    cli.datamodule.setup(stage="fit")
    train_dataloader = cli.datamodule.train_dataloader()
    val_dataloader = cli.datamodule.val_dataloader()
    return train_dataloader, val_dataloader


def cli_main():
    cli = LightningCLI(
        LightningToneClassifier,
        ToneDataModule,
        run=False,
    )

    with open("lightning/trainer_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["trainer"]["logger"] = MLFlowLogger(
        experiment_name="Chinese_tone_classification",
        tracking_uri="http://scoring-api.ponddy.com:8002/",
        log_model=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        min_delta=0.01,
        patience=20,
        verbose=True,
        mode="max",
        check_on_train_epoch_end=False,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="/home/vincent0730/ML_chinese_tone_classification/87/a8",
        filename="{epoch}-{val_loss:.3f}-{val_acc:.3f}",
        monitor="val_acc",
        verbose=True,
        save_top_k=2,
        mode="max",
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    config["trainer"]["callbacks"] = [
        lr_monitor_callback,
        early_stop_callback,
        checkpoint_callback,
    ]
    trainer = Trainer(**config["trainer"])
    trainer.logger.log_hyperparams(config)

    # train_dataloader, val_dataloader = prepare_fit_dataloader(cli)
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    cli_main()
