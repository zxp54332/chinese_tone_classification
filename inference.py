import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from tqdm import tqdm

from baseline_models import CNNet, CustomResNetTextCat
from tone_datamodule import ModuleConfig, ToneDataModule


test_metadata = "/data/tone_speech_cutwav/kaldi_cutwavs_new_2020/test/metadata.csv"
config = ModuleConfig(metadata=test_metadata, n_samples=-1)
tdm = ToneDataModule(config)
tdm.prepare_data(lazy_loading=True)
test_dataloader = tdm.get_lazy_dataloader()

device = "cuda"
# model = CNNet().to(device)

token_mapping = tdm.get_token_mapping()
tdm.max_length_pinyin = 1
tdm.config.text_embed_dim = 64
model = CustomResNetTextCat(
    len(token_mapping), text_embed_dim=tdm.config.text_embed_dim
).to(device)

model.load_state_dict(
    torch.load("resnet_cat_pinyin_adamw/byf1_epoch19_loss0.6367_acc0.78_f10.78.pt")
)
model.eval()

# precision_metric = evaluate.load("precision")
# recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
predict = []
labels = []
with torch.no_grad():
    for data in tqdm(test_dataloader):
        audio_arrays = data["audio_arrays"].to(device)
        label = data["label"].to(device)
        ids = data["pinyin_ids"].to(device)
        ids = torch.flatten(ids)
        offsets = torch.tensor(list(range(0, len(ids), tdm.max_length_pinyin)))
        offsets = offsets.to(device)
        pred = model(audio_arrays, ids, offsets)

        predict.extend(pred.argmax(1).tolist())
        labels.extend(data["label"].tolist())

# confusion matrix
df = pd.DataFrame({"tone": labels, "predict": predict})
array = confusion_matrix(df["tone"], df["predict"])
cf_arrays = [array]

for array in cf_arrays:
    df_cm = (
        pd.DataFrame(
            array,
            index=["0", "1", "2", "3", "4"],
            columns=["0", "1", "2", "3", "4"],
        ),
    )
    fig = plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm[0], annot=True, cmap="Blues_r", fmt="d", cbar=False)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.savefig("resnet_cat_pinyin_adamw.png")

# classification report
target_names = ["Tone 0", "Tone 1", "Tone 2", "Tone 3", "Tone 4"]
cls_report = classification_report(
    labels, predict, target_names=target_names
)
filename = "resnet_cat_pinyin_adamw_report.txt"
with open(filename, "wt", encoding="UTF-8") as f:
    f.write(cls_report)

print(
    f1_metric.compute(references=labels, predictions=predict, average="weighted")["f1"]
)
