import dataclasses
import os
import random
import sys
import warnings

import boto3
import evaluate
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import psutil
import seaborn as sn
import torch
from accelerate import Accelerator
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import baseline_models
from tone_args import get_args_parser
from tone_datamodule import ModuleConfig, ToneDataModule

warnings.filterwarnings("ignore")
NB_PHYSICAL_CORES = psutil.cpu_count(logical=False)

s3_client = boto3.client("s3")
AWS_S3_BUCKET_NAME = "mlflow-models"

f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")

# seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train(
    accelerator, model, dataloader, device, optimizer, cost, use_text=True, tdm=None
):
    model.train()
    train_loss, acc, f1_score = 0, 0, 0
    num_batches = 0

    for data in dataloader:
        num_batches += 1
        audio_arrays = data["audio_arrays"]  # .to(device)
        label = data["label"]  # .to(device)
        if use_text:
            ids = data["pinyin_ids"]  # .to(device)
            ids = torch.flatten(ids)
            offsets = torch.tensor(list(range(0, len(ids), tdm.max_length_pinyin)))
            offsets = offsets.to(device)

        optimizer.zero_grad()
        pred = model(audio_arrays, ids, offsets) if use_text else model(audio_arrays)
        loss = cost(pred, label)
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        train_loss += loss.item()
        predictions = pred.argmax(1)
        acc += accuracy_metric.compute(references=label, predictions=predictions)[
            "accuracy"
        ]
        f1_score += f1_metric.compute(
            references=label, predictions=predictions, average="weighted"
        )["f1"]

    train_loss /= num_batches
    acc /= num_batches
    f1_score /= num_batches
    return train_loss, acc, f1_score


def test(model, dataloader, device, cost, use_text=True, tdm=None):
    model.eval()
    test_loss, acc, f1_score = 0, 0, 0
    num_batches = 0

    with torch.no_grad():
        for data in dataloader:
            num_batches += 1
            audio_arrays = data["audio_arrays"]  # .to(device)
            label = data["label"]  # .to(device)
            if use_text:
                ids = data["pinyin_ids"]  # .to(device)
                ids = torch.flatten(ids)
                offsets = torch.tensor(list(range(0, len(ids), tdm.max_length_pinyin)))
                offsets = offsets.to(device)
                pred = model(audio_arrays, ids, offsets)
            else:
                pred = model(audio_arrays)

            test_loss += cost(pred, label).item()
            predictions = pred.argmax(1)
            acc += accuracy_metric.compute(references=label, predictions=predictions)[
                "accuracy"
            ]
            f1_score += f1_metric.compute(
                references=label, predictions=predictions, average="weighted"
            )["f1"]

    test_loss /= num_batches
    acc /= num_batches
    f1_score /= num_batches
    return test_loss, acc, f1_score


def main(args):
    mixed_precision = None
    if args.mixed_precision:
        mixed_precision = mixed_precision
    accelerator = Accelerator(mixed_precision=mixed_precision)
    # "/data/tone_speech_cutwav/kaldi_cutwavs_new_2020/valid/metadata.csv"
    config = ModuleConfig(metadata=args.train_metadata)
    config.n_samples = args.train_n_samples
    config.batch_size = args.batch_size
    config.audio_feature_type = args.audio_feature
    tdm = ToneDataModule(config)
    

    if args.lazy_loading:
        tdm.prepare_data(lazy_loading=True)
        if args.use_text:
            text_feature_type = args.text_feature
            if text_feature_type == "char":
                # token_mapping = tdm.get_char_mapping()
                sys.exit("Unsupported text feature type. pinyin only.")
            elif text_feature_type == "pinyin":
                token_mapping = tdm.get_token_mapping()
                tdm.max_length_pinyin = 1
                tdm.config.text_embed_dim = 64
            else:
                sys.exit("Unsupported text feature type. pinyin only.")

            train_dataloader = tdm.get_lazy_dataloader(tdm.config.batch_size)
        else:
            sys.exit("lazy_loading always use text.")

    else:
        tdm.prepare_data()
        if args.train_from_cache:
            text_feature_type = args.text_feature
            if text_feature_type == "char":
                token_mapping = tdm.get_char_mapping()
            elif text_feature_type == "pinyin":
                token_mapping = tdm.get_token_mapping()
                tdm.max_length_pinyin = 1
                tdm.config.text_embed_dim = 64
            else:
                sys.exit("Unsupported text feature type. char/pinyin only.")
            tdm.load_data(args.train_from_cache)
        else:
            if args.use_text:
                text_feature_type = args.text_feature
                if text_feature_type == "char":
                    token_mapping = tdm.get_char_mapping()
                elif text_feature_type == "pinyin":
                    token_mapping = tdm.get_token_mapping()
                    tdm.max_length_pinyin = 1
                    tdm.config.text_embed_dim = 64
                else:
                    sys.exit("Unsupported text feature type. char/pinyin only.")
                tdm.preprocess_pinyin_data(feature_type=text_feature_type)
            tdm.preprocess_audio_data()

        train_dataloader = tdm.get_dataloader(batch_size=tdm.config.batch_size)

    # tdm.save_data("train.cache.pinyin")
    # tdm.df.to_csv(f"train{args.train_n_samples}.csv")

    # "/data/tone_speech_cutwav/kaldi_cutwavs_new_2020/test/metadata.csv"
    tdm.config.metadata = args.valid_metadata
    tdm.config.n_samples = args.valid_n_samples
    if args.lazy_loading:
        tdm.prepare_data(lazy_loading=True)
        test_dataloader = tdm.get_lazy_dataloader(tdm.config.batch_size)
    else:
        tdm.prepare_data()
        if args.valid_from_cache:
            tdm.load_data(args.valid_from_cache)
        else:
            if args.use_text:
                text_feature_type = args.text_feature
                tdm.preprocess_pinyin_data(feature_type=text_feature_type)
            tdm.preprocess_audio_data()

        test_dataloader = tdm.get_dataloader()

    # tdm.save_data("valid.cache.pinyin")

    no_text_models = {
        "custom_cnn": baseline_models.CNNet,
        "resnet": baseline_models.CustomResNet,
    }
    text_models = {
        "custom_cnn_cat": baseline_models.CustomTextCNNCat,
        "custom_cnn_sum": baseline_models.CustomTextCNN,
        "resnet_sum": baseline_models.CustomResNetText,
        "resnet_cat": baseline_models.CustomResNetTextCat,
    }
    if args.use_text:
        model = text_models.get(args.model)
    else:
        model = no_text_models.get(args.model)
    if not model:
        sys.exit("Unsupported model type.")
    if args.use_text:
        model = model(len(token_mapping), text_embed_dim=tdm.config.text_embed_dim)
    else:
        model = model()

    # device = args.device
    device = accelerator.device
    model = model.to(device)

    cost = torch.nn.CrossEntropyLoss()
    # https://github.com/pytorch/vision/tree/main/references/classification
    learning_rate = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=learning_rate,
    #     momentum=momentum,
    #     weight_decay=weight_decay,
    # )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        test_dataloader,
    )

    best_loss = float("inf")
    best_loss_checkpoint = ""
    best_f1 = 0
    best_f1_checkpoint = ""
    checkpoint_dir = args.run_name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    # mlflow record
    with mlflow.start_run(experiment_id="87", run_name=args.run_name) as run:
        epochs = args.epochs
        for epoch in tqdm(range(epochs)):
            epoch += 1
            train_loss, train_accuracy, train_f1_score = train(
                accelerator,
                model,
                train_dataloader,
                device,
                optimizer,
                cost,
                args.use_text,
                tdm,
            )
            test_loss, test_accuracy, test_f1_score = test(
                model,
                test_dataloader,
                device,
                cost,
                args.use_text,
                tdm,
            )

            if test_loss < best_loss:
                best_loss = test_loss
                loss_checkpoint = f"{checkpoint_dir}/byloss_epoch{epoch}_loss{test_loss:.4f}_acc{test_accuracy:.2f}_f1{test_f1_score:.2f}.pt"
                if os.path.exists(best_loss_checkpoint):
                    os.remove(best_loss_checkpoint)
                torch.save(model.state_dict(), loss_checkpoint)
                best_loss_checkpoint = loss_checkpoint

            if test_f1_score > best_f1:
                best_f1 = test_f1_score
                f1_checkpoint = f"{checkpoint_dir}/byf1_epoch{epoch}_loss{test_loss:.4f}_acc{test_accuracy:.2f}_f1{test_f1_score:.2f}.pt"
                if os.path.exists(best_f1_checkpoint):
                    os.remove(best_f1_checkpoint)
                torch.save(model.state_dict(), f1_checkpoint)
                best_f1_checkpoint = f1_checkpoint

            mlflow.log_metric("train_loss", train_loss, epoch)
            mlflow.log_metric("test_loss", test_loss, epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, epoch)
            mlflow.log_metric("test_accuracy", test_accuracy, epoch)
            mlflow.log_metric("train_f1_score", train_f1_score, epoch)
            mlflow.log_metric("test_f1_score", test_f1_score, epoch)

        mlflow.log_params(dataclasses.asdict(config))
        mlflow.log_param("epochs", epochs)

        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("best_loss_checkpoint", best_loss_checkpoint)
        mlflow.log_param("best_f1_checkpoint", best_f1_checkpoint)

        # Load best for inference
        for best_checkpoint in [best_loss_checkpoint, best_f1_checkpoint]:
            model.load_state_dict(torch.load(best_checkpoint))
            model = model.to(device)
            model.eval()

            predict = []
            labels = []
            with torch.no_grad():
                for data in test_dataloader:
                    audio_arrays = data["audio_arrays"]  # .to(device)
                    label = data["label"]  # .to(device)
                    if args.use_text:
                        ids = data["pinyin_ids"]  # .to(device)
                        ids = torch.flatten(ids)
                        offsets = torch.tensor(
                            list(range(0, len(ids), tdm.max_length_pinyin))
                        )
                        offsets = offsets.to(device)
                        pred = model(audio_arrays, ids, offsets)
                    else:
                        pred = model(audio_arrays)
                    predict.extend(pred.argmax(1).tolist())
                    labels.extend(label.tolist())

            uri = f"{run.info.experiment_id}/{run.info.run_id}/artifacts"

            # classification report
            target_names = ["Tone 0", "Tone 1", "Tone 2", "Tone 3", "Tone 4"]
            cls_report = classification_report(
                labels, predict, target_names=target_names
            )
            filename = f"{checkpoint_dir}/cls_report.txt"
            with open(filename, "wt", encoding="UTF-8") as f:
                f.write(cls_report)
            s3_client.upload_file(
                filename,
                AWS_S3_BUCKET_NAME,
                f"{uri}/{filename}",
                ExtraArgs={"ContentType": "text/plain", "ACL": "public-read"},
            )

            # confusion matrix
            df = pd.DataFrame({"tone": labels, "predict": predict})
            array = confusion_matrix(df["tone"], df["predict"])
            cf_arrays = [array]
            checkpoint_filename = best_checkpoint.split("/")[-1]
            filename = checkpoint_filename.replace(".pt", ".png")
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
                plt.savefig(filename)

            s3_client.upload_file(
                filename,
                AWS_S3_BUCKET_NAME,
                f"{uri}/{filename}",
                ExtraArgs={"ContentType": "image/png", "ACL": "public-read"},
            )

            s3_client.upload_file(
                best_checkpoint,
                AWS_S3_BUCKET_NAME,
                f"{uri}/{checkpoint_filename}",
                ExtraArgs={"ACL": "public-read"},
            )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
