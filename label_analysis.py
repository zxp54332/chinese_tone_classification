from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections


def main(args):
    metadata = args.csv_file
    df = pd.read_csv(metadata)
    label_dict = df["label"].value_counts().to_dict()
    od_label_dict = collections.OrderedDict(sorted(label_dict.items()))
    item_counts = collections.OrderedDict(sorted(od_label_dict.items()))

    x = np.arange(0, 5)
    print(item_counts.values())
    plt.bar(x, item_counts.values())
    plt.xlabel("Label")
    plt.ylabel("count")
    plt.title("train500000.csv")
    plt.ticklabel_format(useOffset=False)
    plt.tight_layout()
    plt.savefig("train500000.png", dpi=200)


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
