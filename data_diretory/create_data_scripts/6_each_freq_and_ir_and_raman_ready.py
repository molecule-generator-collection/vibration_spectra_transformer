import pandas as pd
import torch
import ast
import os

MAX_LENGTH = 225
HEAVY_SIZE = int(os.environ["HEAVY_SIZE"])

import os
from constants import PROCESSED_DATA_DIRECTORY
os.makedirs(PROCESSED_DATA_DIRECTORY / f"word_level_tokenized_{HEAVY_SIZE}", exist_ok=True)

"""
データを作るにあたってのメモ
1. freqをmax sizeでpadding paddingは-1 (Maxは225)
"""
def pad_list(input_list):
    return input_list + [-1] * (MAX_LENGTH - len(input_list)) if len(input_list) < MAX_LENGTH else input_list

df = pd.read_csv(PROCESSED_DATA_DIRECTORY + f"/molecule_data_raw.csv", index_col=0)
df = df[df["split"] == "test"]
df = df[df["heavy_size"] == HEAVY_SIZE]
freqs = torch.tensor([pad_list(ast.literal_eval(i)) for i in df["freqs"].to_list()])
IRs = torch.tensor([pad_list(ast.literal_eval(i)) for i in df["IRs"].to_list()])
Ramans = torch.tensor([pad_list(ast.literal_eval(i)) for i in df["Ramans"].to_list()])
torch.save(freqs, PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/freqs.pt")
torch.save(IRs, PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/IRs.pt")
torch.save(Ramans, PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/Ramans.pt")
