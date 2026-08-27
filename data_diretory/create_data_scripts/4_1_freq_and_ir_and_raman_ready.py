
import pandas as pd
import torch
import ast
from tqdm import tqdm
from constants import ORIGINAL_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY
import os

HEAVY_SIZE = os.environ["HEAVY_SIZE"]
os.makedirs(PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}", exist_ok=True)
os.makedirs(PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/train", exist_ok=True)
os.makedirs(PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/test", exist_ok=True)
os.makedirs(PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/valid", exist_ok=True)

def pad_list(input_list):
    return input_list + [-1] * (MAX_LENGTH - len(input_list)) if len(input_list) < MAX_LENGTH else input_list

df = pd.read_csv(PROCESSED_DATA_DIRECTORY + f"molecule_data_raw.csv", index_col=0)
for split in tqdm(["train", "test", "valid"]):
    df_split_atom = df[df["split"] == split]
    df_split_atom = df_split_atom[df_split_atom["heavy_size"] == int(HEAVY_SIZE)]
    
    freqs = torch.tensor([pad_list(ast.literal_eval(i)) for i in df_split_atom["freqs"].to_list()])
    IRs = torch.tensor([pad_list(ast.literal_eval(i)) for i in df_split_atom["IRs"].to_list()])
    Ramans = torch.tensor([pad_list(ast.literal_eval(i)) for i in df_split_atom["Ramans"].to_list()])
    torch.save(freqs, PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/{split}/freqs.pt")
    torch.save(IRs, PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/{split}/IRs.pt")
    torch.save(Ramans, PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/{split}/Ramans.pt")
