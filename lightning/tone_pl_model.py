import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, ConfusionMatrix
from transformers import AutoConfig, AutoModelForAudioClassification


class LightningToneClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 5,
        max_length: int = 200,
        hidden_dropout_prob: float = 0.1,
        ignore_mismatched_sizes: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        mode: str = "max",
        factor: float = 0.5,
        patience: int = 2,
        threshold: float = 0.05,
        interval: str = "step",
        frequency: int = 1,
        monitor: str = "val_acc",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
            max_length=max_length,  # 2.0s
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_name_or_path,
            config=self.config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
        # loss function
        self.loss_fct = CrossEntropyLoss()

        # optimizer & scheduler
        self.lr = lr
        self.weight_decay = weight_decay
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

        # metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_labels)
        self.cm = ConfusionMatrix(task="multiclass", num_classes=self.num_labels)

    def forward(self, input_values):
        return self.model(input_values)

    def training_step(self, batch, batch_idx):
        # outputs = self.forward(batch)
        outputs = self.model(**batch)
        train_loss = outputs.loss
        self.log("train_step_loss", train_loss, on_step=True)
        return {"loss": train_loss, "logits": outputs.logits, "labels": batch["labels"]}

    def validation_step(self, batch, batch_idx):
        # outputs = self.forward(batch)
        outputs = self.model(**batch)
        return {"logits": outputs.logits, "labels": batch["labels"]}

    def validation_epoch_end(self, outputs):
        logits = torch.cat([o["logits"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)
        val_acc = self.accuracy(logits, labels)
        val_loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        self.log("val_acc", val_acc, on_epoch=True)
        self.log("val_loss", val_loss, on_epoch=True)

    def training_epoch_end(self, outputs):
        logits = torch.cat([o["logits"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)
        train_acc = self.accuracy(logits, labels)
        train_loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        self.log("train_acc", train_acc, on_epoch=True)
        self.log("train_loss", train_loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
            verbose=True,
        )
        scheduler = {
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": self.interval,
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": self.frequency,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.monitor,
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
        }
        return [optimizer], [scheduler]

    # def test_step(self, test_batch, batch_idx):
    #     if self.args["feature_type"] == "waveform":
    #         x = test_batch["input_values"]
    #         y = test_batch["labels"]
    #     else:
    #         x, y = test_batch

    #     logits = self.forward(x)
    #     _, y_hat = torch.max(logits, dim=1)

    #     test_mat = self.train_confmat(logits, y)

    #     metric_name = "test_acc"
    #     metric_name = "colin_" + metric_name if self.colin else metric_name

    #     self.log(metric_name, self.test_accuracy(logits, y), sync_dist=True)

    #     return {
    #         # "test_acc": test_acc,
    #         "test_mat": test_mat,
    #         "y_true": y,
    #         "y_pred": y_hat,
    #     }

    # def test_epoch_end(self, outputs):
    #     outputs = self.all_gather(outputs)

    #     if self.trainer.is_global_zero:

    #         self._mlflow_log(outputs)

    # def _mlflow_log(self, outputs):
    #     if self.args["accelerator"]:
    #         epoch_val_mat = [
    #             np.sum(x["test_mat"].cpu().detach().numpy(), axis=0) for x in outputs
    #         ]
    #     else:
    #         epoch_val_mat = [x["test_mat"].cpu().detach().numpy() for x in outputs]

    #     epoch_val_mat = np.sum(epoch_val_mat, axis=0)

    #     df_cm = pd.DataFrame(
    #         epoch_val_mat,
    #         index=self.label_list,
    #         columns=self.label_list,
    #     )
    #     confusion_matrix_csv = "test_confusion_matrix.csv"
    #     confusion_matrix_csv = (
    #         "colin_" + confusion_matrix_csv if self.colin else confusion_matrix_csv
    #     )
    #     df_cm.to_csv(confusion_matrix_csv)

    #     plt.figure(figsize=(10, 7))
    #     sn.set(font_scale=1.4)
    #     sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="g")
    #     fig = plt.gcf()
    #     png_file = "test_confusion_matrix.png"
    #     png_file = "colin_" + png_file if self.colin else png_file
    #     plt.savefig(png_file)

    #     mlflow.log_figure(fig, png_file)
    #     mlflow.log_artifact(confusion_matrix_csv)

    #     total_pred = []
    #     total_true = []
    #     for x in outputs:
    #         total_pred.extend(
    #             np.concatenate(x["y_pred"].cpu().detach().numpy(), axis=None)
    #         )
    #         total_true.extend(
    #             np.concatenate(x["y_true"].cpu().detach().numpy(), axis=None)
    #         )
    #     report = classification_report(
    #         total_true,
    #         total_pred,
    #         labels=range(self.args["num_class"]),
    #         output_dict=False,
    #         digits=2,
    #         target_names=self.label_list,
    #     )

    #     test_report = "test_report.txt"
    #     test_report = "colin_" + test_report if self.colin else test_report

    #     with open(test_report, "w", encoding="UTF-8") as file:
    #         file.write(f"Title\n\nClassification Report\n\n{report}")

    #     mlflow.log_artifact(test_report)
