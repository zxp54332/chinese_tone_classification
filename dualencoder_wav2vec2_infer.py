from dualencoder_wav2vec2_bert import (
    VisionTextDualEncoderConfig,
    AudioTextDualEncoderModel,
)
from transformers import AutoModel, AutoFeatureExtractor, AutoTokenizer
import librosa
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import json

df = pd.read_csv("test1000.csv")
config = VisionTextDualEncoderConfig(
    "TencentGameMate/chinese-hubert-base", "distilbert-base-uncased"
)


def get_model():
    vision_model = AutoModel.from_pretrained(
        "TencentGameMate/chinese-hubert-base",
        config=config.vision_config,
        ignore_mismatched_sizes=True,
    )
    text_model = AutoModel.from_pretrained(
        "distilbert-base-uncased", config=config.text_config
    )
    model = AudioTextDualEncoderModel(config, vision_model, text_model)
    model.load_state_dict(torch.load("hubert_distilbert_step6000_test_loss4.1501.pt"))
    return model


model = get_model()
feature_extractor = AutoFeatureExtractor.from_pretrained("TencentGameMate/chinese-hubert-base")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# print(clip_labels) # ['tone-1-dān', 'tone-2-dān', 'tone-3-dān', 'tone-4-dān', 'tone-5-dān']


corrects = 0
error_analysis = {
    "1": {"2": 0, "3": 0, "4": 0, "5": 0},
    "2": {"1": 0, "3": 0, "4": 0, "5": 0},
    "3": {"1": 0, "2": 0, "4": 0, "5": 0},
    "4": {"1": 0, "2": 0, "3": 0, "5": 0},
    "5": {"1": 0, "2": 0, "3": 0, "4": 0},
}

model.eval()
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    # filename = ".../xxx__gop-100__rr-100__pin-de__tone-2----dé.wav"
    filename = row["path"]
    waveform, samplerate = librosa.load(filename, sr=16000)
    image = feature_extractor(waveform, return_tensors="pt")

    pinyin = filename.split("----")[-1].split(".")[0]
    true_label = filename.split("__")[-1].replace("----", "-").replace(".wav", "")
    # print("true_label", true_label)
    clip_labels = [f"tone-{label}-{pinyin}" for label in range(1, 6)]
    tokenized_labels = tokenizer(clip_labels, return_tensors="pt")

    # print(model.get_text_features(**tokenized_labels).shape)
    with torch.no_grad():
        # print("tokenized_labels", tokenized_labels)
        label_emb = model.get_text_features(**tokenized_labels)
        # print(label_emb[0][0]) # tensor(-0.0515) # tensor(0.6798) # tensor(-0.6025)
    label_emb = label_emb.detach().cpu().numpy()

    # normalization
    # print(label_emb.min(), label_emb.max())
    label_emb = label_emb / np.linalg.norm(label_emb, axis=0)
    # print(label_emb.min(), label_emb.max())

    with torch.no_grad():
        img_emb = model.get_image_features(**image).detach().cpu().numpy()
    # print(img_emb.min(), img_emb.max())

    scores = np.dot(img_emb, label_emb.T)
    pred = np.argmax(scores)
    # print(scores, pred)
    # print(clip_labels)
    # print("pred_label", clip_labels[pred])
    sandhi_labels = []
    if "3" in true_label:
        tone2_label = true_label.replace("3", "2")
        sandhi_labels = [true_label, tone2_label]
        # print("sandhi_labels", sandhi_labels)

    # 判斷3聲念2聲
    if sandhi_labels and clip_labels[pred] in sandhi_labels:
        corrects += 1
        continue

    if clip_labels[pred] == true_label:
        corrects += 1
    else:
        error_analysis[true_label[5]][clip_labels[pred][5]] += 1
        # print(error_analysis)

# with open("error_analysis.json", "w", encoding="UTF-8") as f:
#     json.dump(error_analysis, f, ensure_ascii=False, indent=4)

print("accuracy", corrects / df.shape[0])
