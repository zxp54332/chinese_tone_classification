import pandas as pd

df = pd.read_csv("train500000.csv")

tone5_num = df[df["label"] == 4].shape[0]
seed = 42
tone_1_df = df[df["label"] == 0].sample(tone5_num, random_state=seed)
tone_2_df = df[df["label"] == 1].sample(tone5_num, random_state=seed)
tone_3_df = df[df["label"] == 2].sample(tone5_num, random_state=seed)
tone_4_df = df[df["label"] == 3].sample(tone5_num, random_state=seed)
tone_5_df = df[df["label"] == 4]

new_df = pd.concat([tone_1_df, tone_2_df, tone_3_df, tone_4_df, tone_5_df], axis=0)
new_df = new_df.sample(frac=1, random_state=seed).reset_index(drop=True)
new_df.to_csv("train_equal_242415.csv", index=False)
