import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Array3D, Audio, Dataset, Features, load_from_disk, load_dataset
from sklearn.utils import shuffle
from torch.utils.data import Dataset as D
import torch.nn.functional as F


class ToneDataset(D):
    def __init__(self, dataset):
        self.ds = dataset

    def __len__(self):
        return self.ds.shape[0]

    def __getitem__(self, idx):
        return self.ds[idx]


@dataclass
class ModuleConfig:
    metadata: str
    n_samples: int = 10000
    sample_rate: int = 16000
    seed: int = 42
    n_fft: int = 400
    n_mels: int = 128
    n_mfcc: int = 39
    pad_to_sec: float = 0.5
    batch_size: int = 512
    text_embed_dim: int = 4
    audio_feature_type: str = "mel-spec"


class ToneDataModule:
    def __init__(self, config: ModuleConfig):
        self.config = config
        self.df = None
        self.ds = None
        self.char_mapping = {"<pad>": 0}
        self.token_mapping = {"<pad>": 0}
        self.max_length_pinyin = 0
        self.max_length = int(self.config.pad_to_sec * self.config.sample_rate)
        self.num_features = 0

    def get_char_mapping(self):
        max_length_pinyin = 0
        for i, row in self.df.iterrows():
            pinyin = row["path"].split("__pin-")[-1].split("__")[0]
            if len(pinyin) > max_length_pinyin:
                max_length_pinyin = len(pinyin)
            for char in pinyin:
                char = char.lower()
                if char not in self.char_mapping:
                    self.char_mapping[char] = len(self.char_mapping)
        self.max_length_pinyin = max_length_pinyin
        return self.char_mapping

    def get_token_mapping(self):
        for i, row in self.df.iterrows():
            pinyin = row["path"].split("__pin-")[-1].split("__")[0]
            pinyin = pinyin.lower()
            if pinyin not in self.token_mapping:
                self.token_mapping[pinyin] = len(self.token_mapping)
        return self.token_mapping

    def tokenize_pinyin_to_char(self, batch):
        pinyin_ids = []
        for path in batch["path"]:
            pinyin = path.split("__pin-")[-1].split("__")[0]
            ids = [self.char_mapping[p.lower()] for p in pinyin]
            if len(ids) < self.max_length_pinyin:
                ids.extend([0] * (self.max_length_pinyin - len(ids)))
            pinyin_ids.append(ids)
        pinyin_ids = np.stack(pinyin_ids, axis=0)
        batch["pinyin_ids"] = pinyin_ids
        return batch

    def tokenize_pinyin(self, batch):
        pinyin_ids = []
        for path in batch["path"]:
            pinyin = path.split("__pin-")[-1].split("__")[0]
            ids = [self.token_mapping.get(pinyin.lower(), 0)]
            pinyin_ids.append(ids)
        pinyin_ids = np.stack(pinyin_ids, axis=0)
        batch["pinyin_ids"] = pinyin_ids
        return batch

    def waveform_to_mel(self, batch):
        return torchaudio.transforms.MelSpectrogram(
            self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
        )(batch)

    def waveform_to_mfcc(self, batch):
        return torchaudio.transforms.MFCC(self.config.sample_rate, n_mfcc=self.config.n_mfcc)(batch)

    def transform_dataset(self, batch):
        audio_arrays = []
        for x in batch["path"]:
            if x["array"].shape[-1] < self.max_length:
                audio_arrays.append(
                    np.pad(
                        x["array"],
                        (0, self.max_length - x["array"].shape[-1]),
                        "constant",
                    )
                )
            else:
                audio_arrays.append(x["array"][: self.max_length])
        audio_arrays = np.stack(audio_arrays, axis=0)

        if self.config.audio_feature_type == "mel-spec":
            tensors = self.waveform_to_mel(torch.from_numpy(audio_arrays))
        elif self.config.audio_feature_type == "mfcc":
            tensors = self.waveform_to_mfcc(torch.from_numpy(audio_arrays))
        else:
            sys.exit("Unsupported audio feature type.")

        tensors = tensors.unsqueeze(1).repeat(1, 3, 1, 1)
        batch["audio_arrays"] = tensors.numpy()
        return batch

    def prepare_data(self, lazy_loading=False):
        df = pd.read_csv(self.config.metadata)
        if self.config.n_samples == 0:
            sys.exit("n_samples should be greater than 0 or -1(ALL data)")
        if self.config.n_samples > -1:
            df = df.sample(
                n=self.config.n_samples, replace=False, random_state=self.config.seed
            )
        df = shuffle(df, random_state=self.config.seed)
        self.df = df

        if not lazy_loading:
            dataset = Dataset.from_pandas(df)
            dataset = dataset.remove_columns("__index_level_0__")
            self.ds = dataset

    def preprocess_pinyin_data(self, feature_type="char"):
        if feature_type == "char":
            self.ds = self.ds.map(
                self.tokenize_pinyin_to_char,
                batched=True,
                batch_size=self.config.batch_size,
                num_proc=4,
            )
        elif feature_type == "pinyin":
            self.ds = self.ds.map(
                self.tokenize_pinyin,
                batched=True,
                batch_size=self.config.batch_size,
                num_proc=4,
            )

    def preprocess_audio_data(self):
        self.ds = self.ds.cast_column(
            "path", Audio(sampling_rate=self.config.sample_rate)
        )
        if self.config.audio_feature_type == "mel-spec":
            num_features = self.config.n_mels
        elif self.config.audio_feature_type == "mfcc":
            num_features = self.config.n_mfcc
        else:
            sys.exit("Unsupported audio feature type.")
        self.num_features = num_features

        features = Features(
            {
                **self.ds.features,
                "audio_arrays": Array3D(dtype="float32", shape=(3, num_features, 41)),
            }
        )
        ds = self.ds.map(
            self.transform_dataset,
            batched=True,
            batch_size=self.config.batch_size,
            num_proc=4,
            features=features,
        )
        ds = ds.remove_columns("path")
        ds = ds.with_format("numpy")
        self.ds = ds

    def load_data(self, dir):
        self.ds = load_from_disk(dir)

    def save_data(self, dir):
        self.ds.save_to_disk(dir)

    def get_dataloader(self, batch_size=512):
        td = ToneDataset(self.ds)
        return torch.utils.data.DataLoader(td, batch_size=batch_size, pin_memory=True)

    def get_lazy_dataloader(self, batch_size=512):
        dataset = load_dataset(
            "csv", data_files=self.config.metadata, streaming=True, split="train"
        )
        dataset = dataset.with_format("torch")
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, pin_memory=True, collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        tensors, targets, ids = [], [], []
        for data in batch:
            pinyin = data["path"].split("__pin-")[-1].split("__")[0].lower()
            ids.append(self.token_mapping[pinyin])
            targets.append(data["label"])

            waveform, sample_rate = torchaudio.load(filepath=data["path"])
            waveform = torchaudio.transforms.Resample(
                sample_rate, new_freq=self.config.sample_rate
            )(waveform)
            if waveform.shape[-1] < self.max_length:
                waveform = F.pad(
                    waveform, (0, self.max_length - waveform.shape[-1]), "constant"
                )
            tensors.append(waveform[0, : self.max_length])
        tensors = torch.stack(tensors, dim=0)
        tensors = self.waveform_to_mel(tensors).unsqueeze(1).repeat(1, 3, 1, 1)
        return {
            "audio_arrays": tensors,
            "label": torch.tensor(targets),
            "pinyin_ids": torch.tensor(ids).unsqueeze(-1),
        }
