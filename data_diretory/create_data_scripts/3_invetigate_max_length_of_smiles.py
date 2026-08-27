import pandas as pd
import torch
import ast
import os
import sys
import numpy as np
from pathlib import Path


sys.path.append(str((Path(__file__).resolve().parent.parent.parent)))
from utils.tokenizers import SPETokenizerWrapper

# DATA_DIRECTORY = PROCESSED_DATA_DIRECTORY

# smiles_train = torch.load(os.path.join(DATA_DIRECTORY, "train/train_smiles_ids.pt"), map_location="cpu")
# smiles_valid = torch.load(os.path.join(DATA_DIRECTORY, "valid/valid_smiles_ids.pt"), map_location="cpu")
# smiles_test = torch.load(os.path.join(DATA_DIRECTORY, "test/test_smiles_ids.pt"), map_location="cpu")
# smiles_attention_mask_train = torch.load(os.path.join(DATA_DIRECTORY, "train/train_smiles_attention_masks.pt"), map_location="cpu")
# smiles_attention_mask_valid = torch.load(os.path.join(DATA_DIRECTORY, "valid/valid_smiles_attention_masks.pt"), map_location="cpu")
# smiles_attention_mask_test = torch.load(os.path.join(DATA_DIRECTORY, "test/test_smiles_attention_masks.pt"), map_location="cpu")

from constants import PROCESSED_DATA_DIRECTORY
df = pd.read_csv(PROCESSED_DATA_DIRECTORY / "molecule_data_raw.csv", index_col=0)
tokenizer = SPETokenizerWrapper()
tokenized_list = tokenizer.tokenize(df["smiles"].to_list())
tokenized_length_list = [len(tokenized) for tokenized in tokenized_list]
not_tokenized_length_list = [len((smiles)) for smiles in df["smiles"].to_list()]

# print(len(tokenized_list))
print("max tokenized length is")
print(np.max(np.array(tokenized_length_list)))
print("max not tokenized length is")
print(np.max(np.array(not_tokenized_length_list)))
