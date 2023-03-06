import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="PyTorch Tone Classification Training")

    parser.add_argument(
        "--train_metadata",
        default="/data/tone_speech_cutwav/kaldi_cutwavs_new_2020/valid/metadata.csv",
        type=str,
    )
    parser.add_argument(
        "--valid_metadata",
        default="/data/tone_speech_cutwav/kaldi_cutwavs_new_2020/valid/metadata.csv",
        type=str,
    )
    parser.add_argument("--train_n_samples", default=10000, type=int)
    parser.add_argument("--train_from_cache", default="", type=str)
    parser.add_argument("--valid_n_samples", default=1000, type=int)
    parser.add_argument("--valid_from_cache", default="", type=str)
    parser.add_argument("--lazy_loading", action="store_true")
    parser.add_argument(
        "--audio_feature", default="mel-spec", type=str, help="mel-spec or mfcc"
    )
    parser.add_argument("--use_text", action="store_true")
    parser.add_argument(
        "--text_feature", default="char", type=str, help="char or pinyin"
    )
    parser.add_argument("--model", default="custom_cnn_cat", type=str, help="custom_cnn/resnet/custom_cnn_cat/custom_cnn_sum/resnet_sum/resnet_cat")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--mixed_precision", type=str, required=False)
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--run_name", required=True, type=str, help="Name your run."
    )
    parser.add_argument(
        "--lr-scheduler",
        default="steplr",
        type=str,
        help="the lr scheduler (default: steplr)",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=0,
        type=int,
        help="the number of epochs to warmup (default: 0)",
    )
    parser.add_argument(
        "--lr-warmup-method",
        default="constant",
        type=str,
        help="the warmup method (default: constant)",
    )
    parser.add_argument(
        "--lr-warmup-decay", default=0.01, type=float, help="the decay for lr"
    )
    parser.add_argument(
        "--lr-step-size",
        default=30,
        type=int,
        help="decrease lr every step-size epochs",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )

    return parser
