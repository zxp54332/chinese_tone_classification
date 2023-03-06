from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
from pandarallel import pandarallel


def text_extract(row):
    pinyin_no_stress = row["audio"].split("__pin-")[-1].split("__")[0].lower()
    return f"{pinyin_no_stress}[SEP]{row['label']}"


def main(args):
    metadata = args.csv_file
    df = pd.read_csv(metadata)
    df.rename(columns={"path": "audio"}, inplace=True)

    pandarallel.initialize(progress_bar=True)
    df["text"] = df.parallel_apply(text_extract, axis=1)
    df.to_csv("seq2seq10000.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="",
    )
    args = parser.parse_args()
    main(args)
