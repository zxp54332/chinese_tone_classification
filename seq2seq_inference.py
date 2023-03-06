from dataclasses import dataclass

import datasets
import evaluate
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainingArguments,
)

from custom_trainer import CustomSeq2SeqTrainer


features = datasets.Features(
    {
        "path": datasets.Value(dtype="string", id=None),
        "label": datasets.Value(dtype="int64", id=None),
    }
)
stream_dataset = datasets.load_dataset(
    "csv",
    data_files="test1000.csv",
    streaming=True,
    split="train",
    features=features,
)


checkpoint = "seq2seq_test"
model_checkpoint = "seq2seq_test/checkpoint-100000"
feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)
config = AutoConfig.from_pretrained(model_checkpoint)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_checkpoint)

stream_dataset = stream_dataset.cast_column(
    "path", datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
)


model_input_name = feature_extractor.model_input_names[0]


def prepare_dataset(batch):
    sample = batch["path"]
    inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
    batch[model_input_name] = inputs.get(model_input_name)[0]
    batch["input_length"] = len(sample["array"])
    input_str = (
        sample["path"].split("__pin-")[-1].split("__")[0].lower()
        + "[SEP]"
        + str(batch["label"])
    )
    batch["labels"] = tokenizer(input_str).input_ids
    return batch


stream_dataset = stream_dataset.map(prepare_dataset)


acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(pred):
    pred_ids = pred.predictions
    pred_ids = pred_ids.tolist()
    pred_labels = []
    for ids in pred_ids:
        index102 = ids.index(102)
        pred_labels.append(ids[index102 + 1])
    pred_ids = pred_labels

    true_ids = []
    label_ids = pred.label_ids.tolist()
    for ids in label_ids:
        index102 = ids.index(102)
        true_ids.append(ids[index102 + 1])

    acc = acc_metric.compute(predictions=pred_ids, references=true_ids)["accuracy"]
    f1 = f1_metric.compute(
        predictions=pred_ids, references=true_ids, average="weighted"
    )["f1"]
    return {"acc": acc, "f1": f1}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: None
    decoder_start_token_id: int

    def __call__(self, features):
        model_input_name = self.processor.model_input_names[0]
        input_features = [
            {model_input_name: feature[model_input_name]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            encoded_inputs=label_features,
            return_tensors="pt",
            padding="max_length",
            max_length=20,
        )
        batch["decoder_input_ids"] = labels_batch["input_ids"][:, :-1]
        batch["decoder_attention_mask"] = labels_batch["attention_mask"][:, :-1]

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

stream_dataset = stream_dataset.with_format("torch")


trainer_args = Seq2SeqTrainingArguments(
    output_dir="seq2seq_output",
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=2,
    dataloader_drop_last=False,
    no_cuda=True,
)

# 11. Initialize Trainer
trainer = CustomSeq2SeqTrainer(
    model=model,
    args=trainer_args,
    tokenizer=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

test_results = trainer.predict(stream_dataset)
print("test_results_acc", test_results.metrics["test_acc"])
print("test_results_f1", test_results.metrics["test_f1"])
