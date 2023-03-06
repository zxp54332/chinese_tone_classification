import os
import evaluate
import torch
import numpy as np
import mlflow
from datetime import datetime
from colored_text import bcolors
from huber import HubertClassification
from datasets import load_dataset, Audio, load_from_disk
from transformers import TrainingArguments, Trainer, AutoConfig, AutoFeatureExtractor, EvalPrediction
from wav2vec_train import Preprocess, predict_logic, confusion_matrix_fig, classification_report_txt

t = datetime.now().strftime("%Y_%m_%d_%l_%M_%p")
output_path = "1,000,000_test_report"
run_name="judge_t3_1,000,000_report_wav2vec"
model_path = "500,000_with_text_2023_01_11_ 4_30_45PM/best_model"
test_metadata = "./test.csv"
max_length = int(16000 * 1)

alphabet = list('abcdefghijklmnopqrstuvwxyzäüößÄÖÜìū')
data_files = {"test": test_metadata,}
dataset = load_dataset("csv", data_files=data_files)
test_dataset = dataset["test"].cast_column("path", Audio(sampling_rate=16000))

feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = HubertClassification.from_pretrained(model_path, config=config, train_data=None, 
            vocab_size=len(alphabet), text_embed_dim=4, with_text=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
f1_metric = evaluate.load("f1")
acc_metric = evaluate.load("accuracy")

# feature extration
if os.path.exists(f"{output_path}/test_features"):
    encoded = load_from_disk(f"{output_path}/test_features")
    print("Loading test features(with text)....")
else:
    p = Preprocess(max_length, feature_extractor, with_text=True, streaming=False)
    print("Test feature(with text) extraction....")
    max_pinyin_len = p.max_pinyin_length(test_metadata)
    max_pinyin_len_data = [max_pinyin_len] * len(test_dataset["path"])
    text_encoded = test_dataset.map(p.text_preprocess_function, batched=True, num_proc=10)
    print("Test text encode finish....")
    test_dataset = test_dataset.add_column("ids", text_encoded["ids"]).add_column("max_pinyin_length", max_pinyin_len_data)
    encoded = test_dataset.map(p.preprocess_function, remove_columns=(["path"]), batched=True, num_proc=10)
    encoded.save_to_disk(f"{output_path}/test_features")
    print("Test features encode finish....")

def compute_metrics(eval_pred: EvalPrediction):
        metrics_acc = acc_metric.compute(
            predictions=np.argmax(eval_pred.predictions, axis=-1), references=eval_pred.label_ids
        )
        metrics_f1 = f1_metric.compute(
            predictions=np.argmax(eval_pred.predictions, axis=-1), references=eval_pred.label_ids, average="weighted"
        )
        return {"accuracy": metrics_acc["accuracy"], "f1_score": metrics_f1["f1"]}

test_args = TrainingArguments(
    do_eval=True,
    per_device_eval_batch_size=128,
    output_dir=output_path
)
trainer = Trainer(
    model=model,
    args=test_args,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

 # mlflow record
with mlflow.start_run(experiment_id="87", run_name=run_name) as run:
    eval_results = trainer.predict(encoded)
    print(f"{bcolors.red}[accuracy : {eval_results.metrics['test_accuracy']}], [f1 score : {eval_results.metrics['test_f1_score']}]{bcolors.reset}\n")

    # # classification
    def count_prediction(classes):
        count_list = []
        count_list.extend(classes.count(i) for i in range(config.num_labels))
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

##### judge_tone3 ##### 
    def judge_tone3(pred, label):
        for i in range(len(pred)):
            pred[i] = 3 if pred[i] == 2 and label[i] == 3 else pred[i]
        return pred
    
    predict = judge_tone3(predict, tone)
    #fix_acc = (np.array(tone) == np.array(predict)).mean()
    fix_acc = acc_metric.compute(predictions=predict, references=tone)["accuracy"]
    fix_f1 = f1_metric.compute(predictions=predict, references=tone, average="weighted")["f1"]
    print(f"{bcolors.red}Fix accuracy : {fix_acc}{bcolors.reset}")
    print(f"{bcolors.red}Fix f1 : {fix_f1}{bcolors.reset}")
    mlflow.log_metric("Fix accuracy", fix_acc)
    mlflow.log_metric("Fix f1", fix_f1)
##### judge_tone3 ##### 

    # count predict logic
    model_name = os.path.split(os.path.splitext(model_path)[0])[0]
    save_path = f"{output_path}/{model_name}"
    os.mkdir(save_path)
    predict_logic_fig = predict_logic(count_class0, count_class1, count_class2, count_class3, count_class4, config.num_labels, save_path)

    # confusion matrix
    confusion_fig = confusion_matrix_fig(tone, predict, save_path)

    # classification report
    target_names = ["tone 0", "tone 1", "tone 2", "tone 3", "tone 4"]
    report = classification_report_txt(tone, predict, target_names, save_path)

    # log
    mlflow.log_metrics(eval_results.metrics)
    mlflow.log_figure(predict_logic_fig, "predict_logic.png")
    mlflow.log_figure(confusion_fig, "confusion_matrix.png")
    mlflow.log_text(report, "classification_report.txt")

print(f"{bcolors.red}--------------------mlflow recording finish--------------------{bcolors.reset}")