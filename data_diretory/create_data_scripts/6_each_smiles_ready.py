import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.tokenizers import SPETokenizerWrapper
import pandas as pd
import torch
import os
from constants import PROCESSED_DATA_DIRECTORY

HEAVY_SIZE = int(os.environ["HEAVY_SIZE"])

#ディレクトリが存在しない場合は作成
import os
os.makedirs(PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}", exist_ok=True)

tokenizer = SPETokenizerWrapper()
df = pd.read_csv(PROCESSED_DATA_DIRECTORY + f"/molecule_data_raw.csv", index_col=0)
df = df[df["split"] == "test"]
df = df[df["heavy_size"] == HEAVY_SIZE]

print("smiles tokenizing start")
encodings = tokenizer(df["smiles"].to_list())
smiles_ids = encodings.input_ids
attention_masks = encodings.attention_mask

torch.save(smiles_ids, PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/smiles_ids.pt")
torch.save(attention_masks, PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}/smils_attention_masks.pt")
