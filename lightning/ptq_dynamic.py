import os
import time

import torch
from tone_pl_model import LightningToneClassifier
from torchmetrics import Accuracy
from tqdm import tqdm

from tone_datamodule import ToneDataModule

checkpoint = "/home/vincent0730/ML_chinese_tone_classification/87/a8/epoch=0-val_loss=1.250-val_acc=0.521.ckpt"
pl_model = LightningToneClassifier.load_from_checkpoint(checkpoint)
# get the huggingface mdoel from pl_model
model = pl_model.model
model.eval()

# https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/notebooks/bert/Bert-GLUE_OnnxRuntime_quantization.ipynb
# {torch.nn.Linear}
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / (1024 * 1024))
    os.remove("temp.p")


print_size_of_model(model)
print_size_of_model(quantized_model)

torch.save(quantized_model, "quantized_model.pt")
del quantized_model
quantized_model = torch.load("quantized_model.pt")
quantized_model.eval()


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


def model_runtime(model):
    preds = []
    labels = []
    start = time.time()
    for data in tqdm(datamodule.val_dataloader()):
        pred = torch.argmax(model(data["input_values"]).logits, dim=-1).tolist()
        preds.extend(pred)
        labels.extend(data["labels"].tolist())
    print(f"Total time: {time.time() - start}s")
    print("Accuracy:", acc(torch.tensor(preds), torch.tensor(labels)))


model_runtime(model)
model_runtime(quantized_model)
