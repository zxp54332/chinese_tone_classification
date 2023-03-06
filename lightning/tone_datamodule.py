import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoFeatureExtractor


class ToneDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_metadata: str,
        eval_metadata: str,
        model_name_or_path: str,
        audio_column_name: str = "path",
        max_length: int = 200,
        batch_size: int = 32,
        seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_metadata = train_metadata
        self.eval_metadata = eval_metadata
        self.raw_datasets = DatasetDict()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name_or_path,
            return_attention_mask=False,
            max_length=max_length,  # 2.0s
        )
        self.audio_column_name = audio_column_name
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.train_dataset = None
        self.val_dataset = None
        self.seed = seed

    def get_dataset(self, metadata: str, streaming=True):
        features = datasets.Features(
            {
                "path": datasets.Value(dtype="string", id=None),
                "label": datasets.Value(dtype="int64", id=None),
            }
        )
        dataset = load_dataset(
            "csv",
            data_files=metadata,
            streaming=streaming,
            split="train",
            features=features,
        )
        dataset = dataset.cast_column(
            self.audio_column_name,
            datasets.features.Audio(sampling_rate=self.sampling_rate),
        )
        dataset = dataset.map(self.transforms, batched=True, batch_size=self.batch_size, remove_columns=["path", "label"])
        return dataset.with_format("torch")

    def transforms(self, batch):
        samples = [sample["array"] for sample in batch["path"]]
        ast_features = self.feature_extractor(
            samples, sampling_rate=self.sampling_rate, return_tensors="np"
        )
        return {"input_values": ast_features["input_values"], "labels": batch["label"]}

    def prepare_data(self):
        self.raw_datasets["train"] = self.get_dataset(self.train_metadata)
        self.raw_datasets["eval"] = self.get_dataset(self.eval_metadata)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset, self.val_dataset = (
                self.raw_datasets["train"],
                self.raw_datasets["eval"],
            )
            self.train_dataset = self.train_dataset.shuffle(
                seed=self.seed, buffer_size=1000
            )

        if stage == "test":
            # self.test_dataset = None
            raise NotImplementedError()

        if stage == "predict":
            raise NotImplementedError()

    # def collate_fn(self, batch):
    #     input_values = [b["input_values"] for b in batch]
    #     labels = [b["labels"] for b in batch]
    #     input_values = torch.from_numpy(np.stack(input_values)).type(torch.FloatTensor)
    #     labels = torch.from_numpy(np.stack(labels)).type(torch.LongTensor)
    #     return {"input_values": input_values, "labels": labels}

    def train_dataloader(self):
        if self.trainer:
            epoch = self.trainer.current_epoch
            self.train_dataset.set_epoch(epoch)
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            # collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            # collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            # collate_fn=self.collate_fn,
        )
