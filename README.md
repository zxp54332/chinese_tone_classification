<div align="center">

# ML_chinese_tone_classification

</div>

## Environment

- Python 3.8

## Generate metadata.csv

### Original data

- Disclaimer
  1. The data is extracted from the result of GOP force alignment, so it may include low-quality audio, which makes it challenging for the model to be trained.

```txt
# Taipei server 192.168.1.27
/data/tone_speech_cutwav/kaldi_cutwavs_new_2020/train   80%
/data/tone_speech_cutwav/kaldi_cutwavs_new_2020/valid   10%
/data/tone_speech_cutwav/kaldi_cutwavs_new_2020/test    10%
```

### Script to generate original metadata

```bash
# /data/tone_speech_cutwav/kaldi_cutwavs_new_2020/train/metadata.csv
python generate_metadata.py --wav_folder /data/tone_speech_cutwav/kaldi_cutwavs_new_2020/train

# /data/tone_speech_cutwav/kaldi_cutwavs_new_2020/valid/metadata.csv
python generate_metadata.py --wav_folder /data/tone_speech_cutwav/kaldi_cutwavs_new_2020/valid

# /data/tone_speech_cutwav/kaldi_cutwavs_new_2020/test/metadata.csv
python generate_metadata.py --wav_folder /data/tone_speech_cutwav/kaldi_cutwavs_new_2020/test
```

### Results

- We took valid10000.csv for training and test1000.csv for evaluating our baseline model.
- 500000.csv for training and valid10000.csv for evaluating wav2vec, AST model and Dualencoder architecture model.

#### [Baseline model](https://hackmd.io/GY1_3W40QFasiUTkheJ-3g?view#Baseline-Model-Comparison)

```bash
cd /home/vincent0730/ML_chinese_tone_classification
source /venv/bin/activate
source .env

# for usage
python baseline_train.py -h

CUDA_VISIBLE_DEVICES=0 python baseline_train.py \
--train_metadata "/home/vincent0730/ML_chinese_tone_classification/valid10000.csv" \
--valid_metadata "/home/vincent0730/ML_chinese_tone_classification/test1000.csv" \
--train_n_samples 10000 \
--valid_n_samples 1000 \
--audio_feature mfcc \
--use_text \
--text_feature pinyin \
--model custom_cnn_sum \
--device cuda \
--mixed_precision fp16 \
--epochs 200 \
--batch_size 1000 \
--run_name baseline_CustomTextCNNSum_Pinyin_MFCC
```

#### [AST](https://hackmd.io/GY1_3W40QFasiUTkheJ-3g?view#Audio-spectrogram)

```bash
# for usage
python audio_spectrogram.py -h

CUDA_VISIBLE_DEVICES=0 python audio_spectrogram.py \
--train_metadata="train500000.csv" \
--eval_metadata="valid10000.csv" \
--model_name_or_path="MIT/ast-finetuned-audioset-10-10-0.4593" \
--output_dir="audio_spectrogram_test_cosine" \
--do_train \
--do_eval \
--seed=42 \
--audio_column_name="path" \
--ignore_mismatched_sizes \
--max_train_samples=500000 \
--max_eval_samples=10000 \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=64 \
--max_steps=200000 \
--evaluation_strategy="steps" \
--eval_steps=5000 \
--save_steps=5000 \
--fp16 \
--lr_scheduler_type="cosine" \
--learning_rate=3e-4 \
--weight_decay=1e-2 \
--warmup_steps=10000
```

#### [Dualencoder architecture model (AST, DistilBERT)](https://hackmd.io/GY1_3W40QFasiUTkheJ-3g?view#DualEncoder-AST--DistilBERT)

```bash
# for usage
python audio_spectrogram.py -h

CUDA_VISIBLE_DEVICES=0 python audio_spectrogram_text.py \
--train_metadata="train500000.csv" \
--eval_metadata="valid10000.csv" \
--audio_model_name_or_path="MIT/ast-finetuned-audioset-10-10-0.4593" \
--text_model_name_or_path="distilbert-base-uncased" \
--output_dir="dualencoder_test_cosine" \
--do_train \
--do_eval \
--seed=42 \
--audio_column_name="path" \
--ignore_mismatched_sizes \
--max_train_samples=500000 \
--max_eval_samples=10000 \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=32 \
--eval_steps=1500 \
--learning_rate=1e-4 \
--fp16
```

#### [Wav2vec](https://hackmd.io/Gy-8tHoBQwK1I9VwKlbNow#Model-Comparison)

```bash
cd /data2/home/laurence_chen/project/ML_chinese_tone_classification
conda activate chinese_tone
source .env

# ----- training -----
# you need to change certain fields in the wav2vec_train.py
"""
train_data      # training data (*.csv) path
validation_data # validation data (*.csv) path
cache_folder    # save data features
output_dir      # save checkpoint path
mlflow_run_name # name yourself
"""
CUDA_VISIBLE_DEVICES=0 python wav2vec_train.py \
--with_text True \
--streaming False \

# options
-wt, --with_text  (default="True")     # whether to add text to train together
-st, --streaming  (default="False")    # Stream training
-cp, --checkpoint (default=None)

# ----- resume training -----
CUDA_VISIBLE_DEVICES=0 python wav2vec_train.py \
--checkpoint "*_with_text/checkpoint-*"

# ----- inference -----
# you need to change certain fields in the wav2vec_inference.py
"""
output_path    # test report path
run_name       # mlflow run name
model_path     # eval model path
test_metadata  # test data (*.csv) path
"""
CUDA_VISIBLE_DEVICES=0 python wav2vec_inference.py
```
