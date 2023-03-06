import os
import mlflow
import evaluate
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from colored_text import bcolors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from huber import HubertClassification
## 移至 MLflowCallback 的 906 行 on_save 全部註解，如此不會在每個 epoch 上傳 model
from transformers.integrations import MLflowCallback
from transformers import EarlyStoppingCallback
from transformers import AutoFeatureExtractor
from transformers import TrainingArguments, Trainer, EvalPrediction, AutoConfig
from datasets import load_dataset, Audio, load_from_disk
import datasets

train_data = "/home/vincent0730/ML_chinese_tone_classification/train_equal_212415.csv"
validation_data = "valid10000.csv"
cache_folder = "./_cache_data"
output_dir = "./212415_with_text"
model_path = "TencentGameMate/chinese-hubert-base"
#model_path = "500,000_with_text_2023_01_11_ 4_30_45PM/best_model"
mlflow_run_name="21.2415w_with_text"

alphabet = list('abcdefghijklmnopqrstuvwxyzäüößÄÖÜìū')

class Tokenizer:
    def __init__(self, alphabet=None):
        self.alphabet = sorted(list(set(alphabet)))
        self.token_to_idx = {s: [i] for i, s in enumerate(self.alphabet, start=1)}
        self.token_to_idx["<pad>"] = 0
    
    def __call__(self, sentence: str, max_pinyin_len: int) -> list:
        sequence = [self.token_to_idx[c] for c in sentence]
        sequence = [item for items in sequence for item in items]
        if len(sequence) < max_pinyin_len:
            sequence.extend([0] * (max_pinyin_len - len(sequence)))
        return sequence


class Preprocess:
    def __init__(self, max_length, feature_extractor, with_text, streaming):
        self.max_pinyin_len = 0
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.with_text = with_text
        self.streaming = streaming
        
    def text_preprocess_function(self, examples):
        temp_pinyin = ""
        pinyin = []
        tokenizer = Tokenizer(alphabet=alphabet)
        if self.streaming:
            return self.streaming_text_preprocess_function(
                examples, temp_pinyin, pinyin, tokenizer
            )
        for p in examples["path"]:
            index = p["path"].find("pin") + 4
            while p["path"][index] != "_":
                temp_pinyin += p["path"][index]
                index += 1
            pinyin.append(temp_pinyin.lower())
            temp_pinyin = ""
        pinyin_ids = [tokenizer(p, self.max_pinyin_len) for p in pinyin]
        pinyin_ids = np.stack(pinyin_ids, axis=0)
        #print("max_pinyin_length", max_pinyin_len)
        #print("pinyin_ids", pinyin_ids)
        return {"ids": pinyin_ids}

    def streaming_text_preprocess_function(self, examples, temp_pinyin, pinyin, tokenizer):
        p = examples["path"]
        index = p["path"].find("pin") + 4
        while p["path"][index] != "_":
            temp_pinyin += p["path"][index]
            index += 1
        pinyin.append(temp_pinyin.lower())
        temp_pinyin = ""
        pinyin_ids = [tokenizer(p, self.max_pinyin_len) for p in pinyin]
        pinyin_ids = np.stack(pinyin_ids, axis=0)
        examples["ids"] = pinyin_ids
        return examples

    def preprocess_function(self, examples):
        audio_arrays = examples["path"]["array"] if self.streaming else [x["array"] for x in examples["path"]]
        features = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        if self.streaming:
            features = {"input_values": features["input_values"][0], "labels": examples["label"], "ids": examples["ids"], "max_pinyin_length": [self.max_pinyin_len]} \
                    if self.with_text else {"input_values": features["input_values"][0], "labels": examples["label"]}
            #print(features)
            return features
        return features

    def max_pinyin_length(self, csv_files):
        df = pd.read_csv(csv_files)
        for i, row in df.iterrows():
            pinyin = row["path"].split("__pin-")[-1].split("__")[0]
            if len(pinyin) > self.max_pinyin_len:
                self.max_pinyin_len = len(pinyin)
        return self.max_pinyin_len


def train(checkpoint, train_dataset, eval_dataset, num_labels, data_files, with_text, streaming):
    max_length = int(16000 * 1)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    p = Preprocess(max_length, feature_extractor, with_text, streaming)
    if streaming:
        if with_text:
            max_pinyin_len = p.max_pinyin_length(data_files["train"])
            text_encoded = train_dataset.map(p.text_preprocess_function)
            encoded = text_encoded.map(p.preprocess_function, remove_columns=(["path"]))
            max_pinyin_len = p.max_pinyin_length(data_files["validation"])
            text_encoded = eval_dataset.map(p.text_preprocess_function)
            encoded_val = text_encoded.map(p.preprocess_function, remove_columns=(["path"]))
        else:
            encoded = train_dataset.map(p.preprocess_function)
            encoded_val = eval_dataset.map(p.preprocess_function)
        encoded = encoded.with_format("torch")
        encoded_val = encoded_val.with_format("torch")
    else:
        if with_text:
            if os.path.exists(f"{cache_folder}_with_text/train_features"):
                print("Loading training features(with text)....")
                encoded = load_from_disk(f"{cache_folder}_with_text/train_features")
            else:
                print("Training feature(with text) extraction....")
                max_pinyin_len = p.max_pinyin_length(data_files["train"])
                max_pinyin_len_data = [max_pinyin_len] * len(train_dataset["path"])
                text_encoded = train_dataset.map(p.text_preprocess_function, batched=True, num_proc=10)
                print("Training text encode finish....")
                train_dataset = train_dataset.add_column("ids", text_encoded["ids"]).add_column("max_pinyin_length", max_pinyin_len_data)
                encoded = train_dataset.map(p.preprocess_function, remove_columns=(["path"]), batched=True, num_proc=10)
                encoded.save_to_disk(f"{cache_folder}_with_text/train_features") 
            if os.path.exists(f"{cache_folder}_with_text/valid_features"):
                print("Loading validation features(with text)....")
                encoded_val = load_from_disk(f"{cache_folder}_with_text/valid_features")
            else:
                print("Validation feature(with text) extraction....")
                max_pinyin_len = p.max_pinyin_length(data_files["validation"])
                max_pinyin_len_data = [max_pinyin_len] * len(eval_dataset['path'])
                text_encoded = eval_dataset.map(p.text_preprocess_function, batched=True, num_proc=10)
                print("Validation text encode finish....")
                eval_dataset = eval_dataset.add_column("ids",text_encoded["ids"]).add_column("max_pinyin_length", max_pinyin_len_data)
                encoded_val = eval_dataset.map(p.preprocess_function, remove_columns=(["path"]), batched=True, num_proc=10)
                encoded_val.save_to_disk(f"{cache_folder}_with_text/valid_features")
        else:
            if os.path.exists(f"{cache_folder}/train_features"):
                print("Loading training features....")
                encoded = load_from_disk(f"{cache_folder}/train_features")
            else:
                print("Training feature extraction....")
                encoded = train_dataset.map(p.preprocess_function, remove_columns=(["path"]), batched=True, num_proc=10)
                encoded.save_to_disk(f"{cache_folder}/train_features") 
            if os.path.exists(f"{cache_folder}/valid_features"):
                print("Loading validation features")
                encoded_val = load_from_disk(f"{cache_folder}/valid_features")
            else:
                print("Validation feature extraction....")
                encoded_val = eval_dataset.map(p.preprocess_function, remove_columns=(["path"]), batched=True, num_proc=10)
                encoded_val.save_to_disk(f"{cache_folder}/valid_features")

    # load pretrained model
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = num_labels
    model = HubertClassification.from_pretrained(model_path, config=config, train_data=train_data, 
            vocab_size=len(alphabet), text_embed_dim=4, with_text=with_text)
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    def compute_metrics(eval_pred: EvalPrediction):
        metrics_acc = metric_acc.compute(
            predictions=np.argmax(eval_pred.predictions, axis=-1), references=eval_pred.label_ids
        )
        metrics_f1 = metric_f1.compute(
            predictions=np.argmax(eval_pred.predictions, axis=-1), references=eval_pred.label_ids, average="weighted"
        )
        return {"accuracy": metrics_acc["accuracy"], "f1_score": metrics_f1["f1"]}

    # training arguments
    train_batch_size=128
    eval_batch_size=64
    learning_rate=1e-5
    lr_scheduler_type="linear"
    weight_decay=0.005
    save_total_limit=3
    num_train_epochs=35
    ## streaming
    max_steps = 66000
    eval_steps=1500
    save_steps=1500
    

    if streaming:
        training_args = TrainingArguments(
            # no_cuda=True,
            output_dir=output_dir,
            fp16=True,
            gradient_checkpointing=True,
            ignore_data_skip=True,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            max_steps=max_steps,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            weight_decay=weight_decay,
            greater_is_better=True,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
	        save_steps=save_steps,
            save_total_limit=save_total_limit,
            metric_for_best_model="eval_f1_score",
            load_best_model_at_end=True,
        )
    else:
        training_args = TrainingArguments(
            # no_cuda=True,
            output_dir=output_dir,
            fp16=True,
            gradient_checkpointing=True,
            ignore_data_skip=True,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            greater_is_better=True, 
            save_total_limit=save_total_limit,
            metric_for_best_model="eval_f1_score",
            load_best_model_at_end=True,
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded,
        eval_dataset=encoded_val,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    # mlflow record
    with mlflow.start_run(experiment_id="87", run_name=mlflow_run_name) as run:
        if checkpoint is not None:
            trainer.train(resume_from_checkpoint = checkpoint)
        else:
            trainer.train()

        # save best model if load_best_model_at_end=True or save last model
        trainer.save_model(output_dir=f"{output_dir}/best_model")

        # model evaluation
        eval_results = trainer.predict(encoded_val)
        print(f"{bcolors.red}Accuracy：{eval_results.metrics['test_accuracy']}{bcolors.reset}")
        print(f"{bcolors.red}F1 score：{eval_results.metrics['test_f1_score']}{bcolors.reset}")
        # print(eval_results.predictions, eval_results.predictions.shape)

        # # classification
        def count_prediction(classes):
            count_list = []
            for i in range(5):
                count_list.append(classes.count(i))
            return count_list

        predict = []
        tone = []
        class0 = []
        class1 = []
        class2 = []
        class3 = []
        class4 = []
        predict.extend(np.argmax(eval_results.predictions, axis=-1).tolist())
        tone.extend(eval_results.label_ids.tolist())
        predictions = np.argmax(eval_results.predictions, axis=-1)
        #print(predictions)
        for i in range(len(eval_results.label_ids)):
            if eval_results.label_ids[i] == 0:
                class0.append(predictions[i])
            elif eval_results.label_ids[i] == 1:
                class1.append(predictions[i])
            elif eval_results.label_ids[i] == 2:
                class2.append(predictions[i])
            elif eval_results.label_ids[i] == 3:
                class3.append(predictions[i])
            elif eval_results.label_ids[i] == 4:
                class4.append(predictions[i])

        count_class0 = count_prediction(class0)
        count_class1 = count_prediction(class1)
        count_class2 = count_prediction(class2)
        count_class3 = count_prediction(class3)
        count_class4 = count_prediction(class4)

        #print("tone", tone)
        #print("predict", predict)
        #print(eval_results.label_ids)

        # count predict logic
        predict_logic_fig = predict_logic(count_class0, count_class1, count_class2, count_class3, count_class4, num_labels, output_dir)

        # confusion matrix
        confusion_fig = confusion_matrix_fig(tone, predict, output_dir)

        # classification report
        target_names = ["tone 0", "tone 1", "tone 2", "tone 3", "tone 4"]
        report = classification_report_txt(tone, predict, target_names, output_dir)

        # log
        mlflow.log_metrics(eval_results.metrics)
        mlflow.log_figure(predict_logic_fig, "predict_logic.png")
        mlflow.log_figure(confusion_fig, "confusion_matrix.png")
        mlflow.log_text(report, "classification_report.txt")
        MLflowCallback()._ml_flow.pyfunc.log_model(
            "model",
            artifacts={"model_path": f"{output_dir}/best_model"},
            python_model=MLflowCallback()._ml_flow.pyfunc.PythonModel(),
        )
    print(f"{bcolors.green}--------------------mlflow recording finish--------------------{bcolors.reset}")


def predict_logic(class0, class1, class2, class3, class4, num_labels, output_dir):
    fig = plt.figure(figsize=(30, 4))
    x_axis = np.arange(num_labels)
    logic_bar(231, x_axis, class0, "class0")
    logic_bar(232, x_axis, class1, "class1")
    logic_bar(233, x_axis, class2, "class2")
    logic_bar(234, x_axis, class3, "class3")
    logic_bar(235, x_axis, class4, "class4")
    fig.tight_layout()
    plt.savefig(f"{output_dir}/predict_logic.png")
    return fig

def logic_bar(ax, x_axis, count_class, title):
    ax0 = plt.subplot(ax)
    plt.bar(x_axis, count_class)
    for p in ax0.patches:
        ax0.annotate(
            str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.05), fontsize=13
        )
    ax0.set_title(title, fontsize=16)
    ax0.spines["top"].set_visible(False)  # 刪除外框
    ax0.spines["right"].set_visible(False)

def confusion_matrix_fig(tone, predict, output_dir):
    df = pd.DataFrame({"tone": tone, "predict": predict})
    array = confusion_matrix(df["tone"], df["predict"])
    cf_arrays = [array]
    for array in cf_arrays:
        df_cm = (pd.DataFrame(array, index=["0", "1", "2", "3", "4"], columns=["0", "1", "2", "3", "4"]),)
        fig = plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm[0], annot=True, cmap="Blues_r")
        plt.xlabel("Pred")
        plt.ylabel("True")
        plt.savefig(f"{output_dir}/confusion_matrix.png")
    return fig

def classification_report_txt(y_true, y_pred, target_names, output_dir):
    with open (f"{output_dir}/report.txt", "w") as f:
        report = classification_report(y_true, y_pred, target_names=target_names)
        print(report, file = f)
    return report

def main(args, with_text, streaming):
    # load data
    features = datasets.Features(
        {
            "path": datasets.Value(dtype="string", id=None),
            "label": datasets.Value(dtype="int64", id=None),
        }
    )
    data_files = {
            "train": train_data,
            "validation": validation_data,
        }
    dataset = load_dataset(
                "csv",
                data_files=data_files,
                streaming=streaming,
                features=features
            )
    if with_text:
        if not streaming and os.path.exists(f"{cache_folder}_with_text"):
            train_dataset = load_from_disk(f"{cache_folder}_with_text/train")
            eval_dataset = load_from_disk(f"{cache_folder}_with_text/test")
            print("load file(with text) complete")
        else:
            train_dataset = dataset["train"]
            train_dataset = train_dataset.cast_column("path", Audio(sampling_rate=16000))
            eval_dataset = dataset["validation"]
            eval_dataset = eval_dataset.cast_column("path", Audio(sampling_rate=16000))
            if not streaming:
                train_dataset.save_to_disk(f"{cache_folder}_with_text/train/")
                eval_dataset.save_to_disk(f"{cache_folder}_with_text/test/")
                print("save file(with text) complete")
    elif not streaming and os.path.exists(cache_folder):
        train_dataset = load_from_disk(f"{cache_folder}/train")
        eval_dataset = load_from_disk(f"{cache_folder}/test")
        print("load file complete")
    else:
        train_dataset = dataset["train"]
        train_dataset = train_dataset.cast_column("path", Audio(sampling_rate=16000))
        eval_dataset = dataset["validation"]
        eval_dataset = eval_dataset.cast_column("path", Audio(sampling_rate=16000))
        if not streaming:
            train_dataset.save_to_disk(f"{cache_folder}/train/")
            eval_dataset.save_to_disk(f"{cache_folder}/test/")
            print("save file complete")

    # train_label_list = train_dataset.unique("label")
    # train_label_list.sort()  # Let's sort it for determinism
    # eval_label_list = eval_dataset.unique("label")
    # eval_label_list.sort()
    num_labels = 5    #len(train_label_list)
    # print("train_dataset", train_dataset[0]["path"])
    # print("train_labels", train_label_list)
    # print("test_labels", eval_label_list)
    # print(f"A classification problem with {num_labels} classes: {train_label_list}")
    train(args.checkpoint, train_dataset, eval_dataset, num_labels, data_files, with_text, streaming)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint", default=None, type=str, help="path of checkpoint to resume training")
    parser.add_argument("-wt", "--with_text", default="True", type=str, help="wether to add text training")
    parser.add_argument("-st", "--streaming", default="False", type=str, help="for large csv files use")
    args = parser.parse_args()
    with_text = args.with_text == "True"
    streaming = args.streaming == "True"
    main(args, with_text, streaming)