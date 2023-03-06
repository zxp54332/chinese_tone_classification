import torch
from neural_compressor.config import (
    AccuracyCriterion,
    PostTrainingQuantConfig,
    TuningCriterion,
)
from neural_compressor.quantization import fit
from neural_compressor.utils.pytorch import load
from tone_pl_model import LightningToneClassifier
from torchmetrics import Accuracy
from tqdm import tqdm
from transformers import ASTForAudioClassification

from tone_datamodule import ToneDataModule

accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)
tuning_criterion = TuningCriterion(max_trials=1)
conf = PostTrainingQuantConfig(
    approach="dynamic",
    # backend="default",
    tuning_criterion=tuning_criterion,
    accuracy_criterion=accuracy_criterion,
)

checkpoint = "/home/vincent0730/ML_chinese_tone_classification/87/a8/epoch=0-val_loss=1.250-val_acc=0.521.ckpt"
pl_model = LightningToneClassifier.load_from_checkpoint(checkpoint)
# get the huggingface mdoel from pl_model
model = pl_model.model
model.save_pretrained("ast_fp32")
model.eval()

acc = Accuracy(task="multiclass", num_classes=5)
eval_metadata = "/home/vincent0730/ML_chinese_tone_classification/test1000.csv"
max_length = 75
batch_size = 128
datamodule = ToneDataModule(
    train_metadata="",
    eval_metadata=eval_metadata,
    model_name_or_path="MIT/ast-finetuned-audioset-10-10-0.4593",
    max_length=max_length,
    batch_size=batch_size,
)
valid_dataset = datamodule.get_dataset(datamodule.eval_metadata, streaming=False)
datamodule.val_dataset = valid_dataset


def eval_func(model):
    preds = []
    labels = []
    for data in tqdm(datamodule.val_dataloader()):
        pred = torch.argmax(model(data["input_values"]).logits, dim=-1).tolist()
        preds.extend(pred)
        labels.extend(data["labels"].tolist())
    return acc(torch.tensor(preds), torch.tensor(labels)).item()


q_model = fit(
    model=model,
    conf=conf,
    eval_func=eval_func,
)
q_model.save("./output")

# Testing
x = torch.randn(1, 75, 128)
print(model(x))
print(q_model(x))
del q_model

# https://github.com/intel/neural-compressor/blob/master/neural_compressor/utils/load_huggingface.py
model = ASTForAudioClassification.from_pretrained("ast_fp32")
q_model = load("output/best_model.pt", model)
print(q_model(x))
