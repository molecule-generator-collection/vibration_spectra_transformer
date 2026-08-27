
import pandas as pd
import torch
import ast
from constants import PROCESSED_DATA_DIRECTORY

df = pd.read_csv(PROCESSED_DATA_DIRECTORY + "/molecule_data_raw.csv", index_col=0)
df = df[df["freqs"].map(lambda x: len(ast.literal_eval(x))) <= 100]
df.to_csv(PROCESSED_DATA_DIRECTORY + "/molecule_data_raw.csv")
