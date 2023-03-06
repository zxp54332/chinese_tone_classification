import logging
import os
import sys
import warnings
import dataclasses
from dataclasses import dataclass, field
from random import randint
from typing import Optional
from tqdm import tqdm

import mlflow
import datasets
import evaluate
import numpy as np
import transformers
import torch
import numpy as np
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from accelerate import Accelerator

from dualencoder_wav2vec2_bert import AudioTextDualEncoderModel, VisionTextDualEncoderConfig


logger = logging.getLogger(__name__)



def collate_fn(batch):
    input_values, input_ids, attention_mask = [], [], []
    for b in batch:
        input_values.append(b["input_values"])
        input_ids.append(b["input_ids"])
        attention_mask.append(b["attention_mask"])

    input_values = torch.from_numpy(np.stack(input_values)).type(torch.FloatTensor)
    input_ids = torch.from_numpy(np.stack(input_ids)).type(torch.LongTensor)
    attention_mask = torch.from_numpy(np.stack(attention_mask)).type(torch.FloatTensor)
    return {
        "input_values": input_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A file containing the training audio paths and labels."},
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "A file containing the validation audio paths and labels."},
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to 'validation'"
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    label_column_name: str = field(
        default="label",
        metadata={
            "help": "The name of the dataset column containing the labels. Defaults to 'label'"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_length_seconds: float = field(
        default=20,
        metadata={
            "help": "Audio clips will be randomly cut to this length during training if the value is set."
        },
    )
    train_metadata: str = field(default=None, metadata={"help": "Train metadata."})
    eval_metadata: str = field(default=None, metadata={"help": "Eval metadata."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    audio_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    audio_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    text_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    text_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from the Hub"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    attention_mask: bool = field(
        default=True,
        metadata={
            "help": "Whether to generate an attention mask in the feature extractor."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to freeze the feature extractor layers of the model."
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )

    def __post_init__(self):
        if not self.freeze_feature_extractor and self.freeze_feature_encoder:
            warnings.warn(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "will be removed in a future version. Use `--freeze_feature_encoder`"
                "instead. Setting `freeze_feature_encoder==True`.",
                FutureWarning,
            )
        if self.freeze_feature_extractor and not self.freeze_feature_encoder:
            raise ValueError(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "should not be used in combination with `--freeze_feature_encoder`."
                "Only make use of `--freeze_feature_encoder`."
            )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_audio_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    features = datasets.Features(
        {
            "path": datasets.Value(dtype="string", id=None),
            "label": datasets.Value(dtype="int64", id=None),
        }
    )
    # Initialize our dataset and prepare it for the audio classification task.
    raw_datasets = DatasetDict()
    raw_datasets["train"] = load_dataset(
        "csv",
        data_files=data_args.train_metadata,
        streaming=True,
        split="train",
        features=features,
    )
    raw_datasets["eval"] = load_dataset(
        "csv",
        data_files=data_args.eval_metadata,
        streaming=True,
        split="train",
        features=features,
    )

    # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
    # transformer outputs in the classifier, but it doesn't always lead to better accuracy
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.audio_model_name_or_path,
        return_attention_mask=False,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.text_model_name_or_path)

    # `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets["train"] = raw_datasets["train"].cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )
    raw_datasets["eval"] = raw_datasets["eval"].cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    def train_transforms(batch):
        """Apply train_transforms across a batch."""
        sample = batch["path"]
        text = "-".join(sample["path"].split("__")[-1].split("----")).replace(
            ".wav", ""
        )
        encoded_text = tokenizer(
            text, padding="max_length", max_length=32, truncation=True
        )
        encoded_features = feature_extractor(
            sample["array"],
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(16000*1.0),
            padding="max_length",
            truncation=True,
        )
        array = encoded_features["input_values"][0]
        return {
            "input_values": array,
            "input_ids": encoded_text["input_ids"],
            "attention_mask": encoded_text["attention_mask"],
        }

    config = VisionTextDualEncoderConfig(
        model_args.audio_model_name_or_path, model_args.text_model_name_or_path
    )
    vision_model = AutoModel.from_pretrained(
        model_args.audio_model_name_or_path,
        config=config.vision_config,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    text_model = AutoModel.from_pretrained(
        model_args.text_model_name_or_path, config=config.text_config
    )
    model = AudioTextDualEncoderModel(config, vision_model, text_model)

    # freeze the convolutional waveform encoder
    # if model_args.freeze_feature_encoder:
    #     model.freeze_feature_encoder()

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].shuffle(
                seed=training_args.seed
            )
        # Set the training transforms
        raw_datasets["train"] = raw_datasets["train"].map(train_transforms, remove_columns=["path", "label"])

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = raw_datasets["eval"].shuffle(seed=training_args.seed)
        # Set the validation transforms
        raw_datasets["eval"] = raw_datasets["eval"].map(train_transforms, remove_columns=["path", "label"])

    train_dataloader = DataLoader(raw_datasets["train"].with_format("torch"), batch_size=training_args.per_device_train_batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(raw_datasets["eval"].with_format("torch"), batch_size=training_args.per_device_eval_batch_size, collate_fn=collate_fn)
    
    mixed_precision = None
    if training_args.fp16:
        mixed_precision = "fp16"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        test_dataloader,
    )

    def evaluate():
        ## eval
        model.eval()
        test_loss = 0
        num_test_steps = 0
        for data in test_dataloader:
            num_test_steps += 1
            with torch.no_grad():
                clip_output = model(**data, return_loss=True)
            loss = clip_output.loss
            test_loss += loss.item()
        test_loss /= num_test_steps
        print("test_loss", test_loss)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), f"hubert_distilbert_step{num_steps}_test_loss{test_loss:.4f}.pt")
        return test_loss


    with mlflow.start_run(experiment_id="87", run_name="hubert_bert_dualencoder") as run:
        mlflow.log_params(dataclasses.asdict(model_args))
        mlflow.log_params(dataclasses.asdict(data_args))
        mlflow.log_params(dataclasses.asdict(training_args))
        # train
        train_loss = 0
        num_steps = 0
        for data in tqdm(train_dataloader):
            model.train()
            num_steps += 1
            optimizer.zero_grad()
            clip_output = model(**data, return_loss=True)
            loss = clip_output.loss
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item()

            if not num_steps % training_args.eval_steps:
                train_loss /= num_steps
                test_loss = evaluate()
                mlflow.log_metric("train_loss", train_loss, num_steps)
                mlflow.log_metric("test_loss", test_loss, num_steps)


if __name__ == "__main__":
    main()
