import os
import time
import torch
import numpy as np
# torch 請放在 onnx 前面，避免無法使用CUDA
import onnx
import onnxruntime
from onnxruntime.quantization import QuantType, quantize_dynamic
from tone_pl_model import LightningToneClassifier
from torchmetrics import Accuracy
from tqdm import tqdm

from tone_datamodule import ToneDataModule

checkpoint = "/home/vincent0730/ML_chinese_tone_classification/87/a8/epoch=0-val_loss=1.250-val_acc=0.521.ckpt"
pl_model = LightningToneClassifier.load_from_checkpoint(checkpoint)
model = pl_model

# 輸出成 onnx 格式
# 這裡要注意的是 dynamic_axes，如果輸入的維度是動態的，在這邊要進行設定
# 這裡我們設定 第0維 batch_size 跟第1維 waveform，是動態的
x = torch.randn(1, 75, 128)
model_name = "tone.onnx"
model.to_onnx(
    model_name,
    x,
    export_params=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size", 1: "length"}, "output": {0: "batch_size"}},
)

# ============================測試 ONNX format 模型============================
# 載入輸出好的onnx模型，模型大小為 3xx mb
# https://github.com/microsoft/onnxruntime/issues/11092
# https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#python
providers = [
    # ('CUDAExecutionProvider', {
    #     'device_id': 1,
    # }),
    'CPUExecutionProvider',
]
ort_session = onnxruntime.InferenceSession(
    model_name, providers=providers
)
onnx_model = onnx.load(model_name)
onnx.checker.check_model(onnx_model)
print("name", ort_session.get_inputs()[0].name)

# 這裡的 'input' 對應到上面的 `input_names`
ort_inputs = {"input": torch.randn(1, 75, 128).numpy()}
ort_outs = ort_session.run(None, ort_inputs)
print("ort_outs", ort_outs[0])

# https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/notebooks/bert/Bert-GLUE_OnnxRuntime_quantization.ipynb
# https://github.com/microsoft/onnxruntime/issues/3130#issuecomment-1105200621
# https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu
# Due to a limitation of ONNX Runtime, it is not possible to run quantized models on CUDAExecutionProvider
def quantize_onnx_model(onnx_model_path, quantized_model_path):
    quantize_dynamic(
        onnx_model_path, quantized_model_path, weight_type=QuantType.QUInt8
    )
    print(f"quantized model saved to:{quantized_model_path}")


quantize_onnx_model(model_name, f"quantized_{model_name}")
print(
    "ONNX full precision model size (MB):", os.path.getsize(model_name) / (1024 * 1024)
)
print(
    "ONNX quantized model size (MB):",
    os.path.getsize(f"quantized_{model_name}") / (1024 * 1024),
)

acc = Accuracy(task="multiclass", num_classes=5)


def get_val_dataloader():
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
    val_dataloader = datamodule.val_dataloader()
    return val_dataloader


def onnx_runtime(val_dataloader, ort_session):
    preds = []
    labels = []
    start = time.time()
    for data in tqdm(val_dataloader):
        ort_inputs = {"input": data["input_values"].numpy()}
        ort_outs = ort_session.run(None, ort_inputs)[0]
        pred = np.argmax(ort_outs, axis=-1).tolist()
        preds.extend(pred)
        labels.extend(data["labels"].tolist())
    print(f"Total time: {time.time() - start}s")
    print("Accuracy:", acc(torch.tensor(preds), torch.tensor(labels)))


val_dataloader = get_val_dataloader()
for model in [model_name, f"quantized_{model_name}"]:
    session = onnxruntime.InferenceSession(
        model, providers=providers
    )
    # warm-up run
    session.run(None, ort_inputs)
    onnx_runtime(val_dataloader, session)
