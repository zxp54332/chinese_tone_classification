import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
from pandarallel import pandarallel
from tqdm import tqdm
import time

tqdm.pandas()
df = pd.read_csv("train500000.csv")


def calc_length(r):
    waveform, rate = sf.read(r["path"])
    return waveform.shape[-1] / rate


pandarallel.initialize(progress_bar=True)
start = time.time()
df["length"] = df.parallel_apply(calc_length, axis=1)
print("Elapsed time:", time.time() - start)
df.to_csv("train500000_length.csv", index=False)

df = pd.read_csv("train500000_length.csv")
print(f'Max length in valid: {max(df["length"])}')
plt.title("train500000.csv")
plt.xlabel("Length")
plt.ylabel("Count")
plt.grid()
plt.hist(df["length"])
plt.tight_layout()
plt.savefig("train500000_length.png")
