import os

import json
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import multiprocessing as mp



import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm




BASE_DIR = 'SILFA' 
train = pd.read_csv('train.csv')

train.head()

example_fn = train.query('sign == "T"')["path"].values[0]

example_landmark = pd.read_parquet(f"./{example_fn}")
example_landmark.head()
print(example_landmark)