import os
from argparse import ArgumentParser

import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm


def main(args):
    result = {"path": [], "label": []}
    wav_folder = args.wav_folder
    # new_folder = os.path.join(wav_folder, "16k_mono")
    # os.makedirs(new_folder, exist_ok=True)

    # for i in range(1, 6):
    #     tone_folder = os.path.join(new_folder, f"{i}")
    #     os.makedirs(tone_folder, exist_ok=True)

    for subfolder in tqdm(os.listdir(wav_folder)):
        for i in tqdm(range(1, 6)):
            tone_folder = os.path.join(wav_folder, subfolder, f"{i}")
            for file in tqdm(os.listdir(tone_folder)):
                abs_file = os.path.join(tone_folder, file)
                # segment = AudioSegment.from_file(abs_file)
                # segment = segment.set_channels(1)
                # segment = segment.set_frame_rate(16000)
                # resample_file = segment.export(
                #     f"{os.path.join(new_folder, f'{i}', file)}", format="wav"
                # ).name
                result["path"].append(abs_file)
                result["label"].append(i - 1)

    df = pd.DataFrame(result)
    df.to_csv(f"{wav_folder}/metadata.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument(
        "--wav_folder",
        type=str,
        required=True,
        help="",
    )
    args = parser.parse_args()
    main(args)
