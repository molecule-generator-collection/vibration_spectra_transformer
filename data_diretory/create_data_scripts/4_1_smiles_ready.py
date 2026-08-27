import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.tokenizers import SPETokenizerWrapper
import pandas as pd
import torch
from tqdm import tqdm
from constants import ORIGINAL_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY

import os
HEAVY_SIZE = os.environ["HEAVY_SIZE"]
os.makedirs(PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}", exist_ok=True)
os.makedirs(PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/train", exist_ok=True)
os.makedirs(PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/test", exist_ok=True)
os.makedirs(PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/valid", exist_ok=True)

tokenizer = SPETokenizerWrapper()
df = pd.read_csv(PROCESSED_DATA_DIRECTORY + f"/molecule_data_raw.csv", index_col=0)

print("smiles tokenizing start")
for split in tqdm(["train", "test", "valid"]):
    df_split_atom = df[df["split"] == split]
    df_split_atom = df_split_atom[df_split_atom["heavy_size"] == int(HEAVY_SIZE)]

    encodings = tokenizer(df_split_atom["smiles"].to_list())
    smiles_ids = encodings.input_ids
    attention_masks = encodings.attention_mask

    torch.save(smiles_ids, PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/{split}/smiles_ids.pt")
    torch.save(attention_masks, PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/{split}/smils_attention_masks.pt")
