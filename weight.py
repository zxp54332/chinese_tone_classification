import pandas as pd
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight 
from colored_text import bcolors

def weight_balance(train_data):
    print(f"{bcolors.red}weight balance calculation for training data：{train_data}{bcolors.reset}")
    y = []
    data = pd.read_csv(train_data)
    y.extend(data.iat[i, 1] for i in range(len(data)))
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    weights = torch.tensor(weights,dtype=torch.float)
    print(f"{bcolors.red}class_weight：{weights}{bcolors.reset}")
    return weights